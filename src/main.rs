use nalgebra::Isometry3;
use serde::Deserialize;
use std::{io::Read, net::TcpStream, sync::mpsc::channel, thread::{sleep, spawn}, time::{Duration, Instant}};
use franka::{Frame, MotionFinished};
use franka::FrankaResult;
use franka::Robot;
use franka::RobotState;
use franka::Torques;
use franka::{array_to_isometry, Matrix6x7, Vector7};
use nalgebra::{Matrix3, Matrix6, Matrix6x1, UnitQuaternion, Vector3, U1, U3};

#[derive(Debug, Deserialize)]
struct PoseData {
    quaternion: Quaternion,
    translation: Translation,
}

#[derive(Debug, Deserialize)]
struct Quaternion {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

#[derive(Debug, Deserialize)]
struct Translation {
    x: f64,
    y: f64,
    z: f64,
}

fn main() {
    let (tx, rx) = channel();
    let mut tcp_stream = TcpStream::connect("192.168.5.9:8080").unwrap();
    println!("Connected to the server!");
    let mut v = vec![0; 512];
    let mut last_pose = nalgebra::Isometry3::identity();
    spawn(||{
        robot_control(rx);
    });
    loop {
        let size = tcp_stream.read(&mut v).unwrap();
        let pose: PoseData = serde_json::from_slice(&v[0..size]).unwrap();
        let t =
            nalgebra::Translation3::new(pose.translation.x, pose.translation.y, pose.translation.z);
        let q = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new (
            pose.quaternion.w,
            pose.quaternion.x,
            pose.quaternion.y,
            pose.quaternion.z,
        ));
        let pose = nalgebra::Isometry3::from_parts(t, q);
        let change = last_pose.inv_mul(&pose);
        last_pose = pose;
        tx.send(change).unwrap();
    }
}


fn robot_control_(franka_ip: String, rx: std::sync::mpsc::Receiver<Isometry3<f64>>) -> FrankaResult<()> {
    let mut last_change = nalgebra::Isometry3::identity();
    let translational_stiffness = 150.;
    let rotational_stiffness = 10.;

    let mut stiffness: Matrix6<f64> = Matrix6::zeros();
    let mut damping: Matrix6<f64> = Matrix6::zeros();
    {
        let mut top_left_corner = stiffness.fixed_view_mut::<3, 3>(0, 0);
        top_left_corner.copy_from(&(Matrix3::identity() * translational_stiffness));
        let mut top_left_corner = damping.fixed_view_mut::<3, 3>(0, 0);
        top_left_corner.copy_from(&(2. * f64::sqrt(translational_stiffness) * Matrix3::identity()));
    }
    {
        let mut bottom_right_corner = stiffness.fixed_view_mut::<3, 3>(3, 3);
        bottom_right_corner.copy_from(&(Matrix3::identity() * rotational_stiffness));
        let mut bottom_right_corner = damping.fixed_view_mut::<3, 3>(3, 3);
        bottom_right_corner
            .copy_from(&(2. * f64::sqrt(rotational_stiffness) * Matrix3::identity()));
    }
    let mut robot = Robot::new(&franka_ip, None, None)?;
    let model = robot.load_model(true)?;

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.set_collision_behavior(
        [100.; 7], [100.; 7], [100.; 7], [100.; 7], [100.; 6], [100.; 6], [100.; 6], [100.; 6],
    )?;
    robot.set_joint_impedance([3000., 3000., 3000., 2500., 2500., 2000., 2000.])?;
    robot.set_cartesian_impedance([3000., 3000., 3000., 300., 300., 300.])?;
    let initial_state = robot.read_once()?;
    let initial_transform = array_to_isometry(&initial_state.O_T_EE);
    let position_d = initial_transform.translation.vector;
    let orientation_d = initial_transform.rotation;

    println!(
        "WARNING: Collision thresholds are set to high values. \
             Make sure you have the user stop at hand!"
    );
    println!("After starting try to push the robot and see how it reacts.");
    println!("Press Enter to continue...");
    std::io::stdin().read_line(&mut String::new()).unwrap();
    let result = robot.control_torques(
        |state: &RobotState, _step: &Duration| -> Torques {
            let change = match rx.try_recv() {
                Ok(p) => {
                    last_change = p;
                    p
                }
                Err(e) => {
                    if e == std::sync::mpsc::TryRecvError::Empty {
                        last_change
                    } else {
                        return Torques::new([0., 0., 0., 0., 0., 0., 0.]).motion_finished();
                    }
                }
            };

            let coriolis: Vector7 = model.coriolis_from_state(&state).into();
            let jacobian_array = model.zero_jacobian_from_state(&Frame::EndEffector, &state);
            let jacobian = Matrix6x7::from_column_slice(&jacobian_array);
            let _q = Vector7::from_column_slice(&state.q);
            let dq = Vector7::from_column_slice(&state.dq);
            let current_transform = array_to_isometry(&state.O_T_EE);
            let rot = nalgebra::Rotation3::<f64>::from_matrix(
                &Matrix4::from_column_slice(&state.O_T_EE)
                    .remove_column(3)
                    .remove_row(3),
            );
            let current_transform = Isometry3::from_parts(
                Vector3::new(state.O_T_EE[12], state.O_T_EE[13], state.O_T_EE[14]).into(),
                rot.into(),
            );
            let transform = current_transform * last_change;
            
            let position = transform.translation.vector;
            let mut orientation = *transform.rotation.quaternion();

            let mut error: Matrix6x1<f64> = Matrix6x1::<f64>::zeros();
            {
                let mut error_head = error.fixed_slice_mut::<U3, U1>(0, 0);
                error_head.set_column(0, &(position - position_d));
            }

            if orientation_d.coords.dot(&orientation.coords) < 0. {
                orientation.coords = -orientation.coords;
            }
            let orientation = UnitQuaternion::new_normalize(orientation);
            let error_quaternion: UnitQuaternion<f64> = orientation.inverse() * orientation_d;
            {
                let mut error_tail = error.fixed_slice_mut::<U3, U1>(3, 0);
                error_tail.copy_from(
                    &-(transform.rotation.to_rotation_matrix()
                        * Vector3::new(error_quaternion.i, error_quaternion.j, error_quaternion.k)),
                );
            }
            let tau_task: Vector7 =
                jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
            let tau_d: Vector7 = tau_task + coriolis;

            tau_d.into()
        },
        None,
        None,
    );

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("{}", e);
            Ok(())
        }
    }
}
