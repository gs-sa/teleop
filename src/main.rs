use franka::FrankaResult;
use franka::Robot;
use franka::RobotState;
use franka::Torques;
use franka::{Frame, MotionFinished};
use nalgebra::Matrix6;
use nalgebra::Quaternion as NaQuaternion;
use nalgebra::Rotation3;
use nalgebra::Translation3;
use nalgebra::{Isometry3, Matrix4, SMatrix};
use nalgebra::{Matrix3, Matrix6x1, UnitQuaternion, Vector3};
use serde::Deserialize;
use tungstenite::connect;
use std::{sync::mpsc::channel, thread::spawn, time::Duration};

#[derive(Debug, Deserialize)]
struct PoseData {
    quaternion: Quaternion,
    translation: Translation,
}

impl PoseData {
    fn to_isometry(&self) -> Isometry3<f64> {
        Isometry3::from_parts(Translation3::new(self.translation.x, self.translation.y, self.translation.z), UnitQuaternion::from_quaternion(NaQuaternion::new(self.quaternion.w, self.quaternion.x, self.quaternion.y, self.quaternion.z)))
    }
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
    let (mut ws, _) = connect("ws://192.168.5.12:8080").unwrap();
    println!("Connected to the server!");

    let m_to_mee = Isometry3::from_parts(
        Translation3::identity(), 
        Rotation3::from_matrix(&Matrix3::from_column_slice(&[
            0.,1.,0.,
            1.,0.,0.,
            0.,0.,-1.
        ])).into()
    );
    let mut mo_t_m_prev = None;
    spawn(|| {
        robot_control("192.168.1.100".to_owned(), rx).unwrap();
    });
    loop {
        let msg = ws.read().unwrap();
        let pose: PoseData = serde_json::from_slice(&msg.into_data()).unwrap();
        let mo_t_m = pose.to_isometry();
        let delta = mo_t_m_prev.map(|mo_t_m_prev: Isometry3<f64>|{
            let mprev_t_m = mo_t_m_prev.inv_mul(&mo_t_m);
            m_to_mee.inv_mul(&(mprev_t_m * m_to_mee))
        });
        mo_t_m_prev = Some(mo_t_m);
        if delta.is_some() {tx.send(delta.unwrap()).unwrap()}
    }
}

fn stiffness_damping(translational_stiffness: f64, rotational_stiffness: f64) -> (Matrix6<f64>, Matrix6<f64>){
    let mut stiffness = SMatrix::<f64, 6, 6>::zeros();
    let mut damping = SMatrix::<f64, 6, 6>::zeros();
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
    (stiffness, damping)
}

fn robot_control(
    franka_ip: String,
    rx: std::sync::mpsc::Receiver<Isometry3<f64>>,
) -> FrankaResult<()> {
    // start config

    let (stiffness, damping) = stiffness_damping(300., 20.);
    
    let mut robot = Robot::new(&franka_ip, Some(franka::RealtimeConfig::Ignore), None)?;
    let model = robot.load_model(true)?;

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.set_collision_behavior(
        [100.; 7], [100.; 7], [100.; 7], [100.; 7], [100.; 6], [100.; 6], [100.; 6], [100.; 6],
    )?;
    robot.set_joint_impedance([3000., 3000., 3000., 2500., 2500., 2000., 2000.])?;
    robot.set_cartesian_impedance([3000., 3000., 3000., 300., 300., 300.])?;
    
    println!(
        "WARNING: Collision thresholds are set to high values. \
             Make sure you have the user stop at hand!"
    );
    println!("After starting try to push the robot and see how it reacts.");
    println!("Press Enter to continue...");
    std::io::stdin().read_line(&mut String::new()).unwrap();
    // end config

    let state = robot.read_once().unwrap();
    let rot = nalgebra::Rotation3::<f64>::from_matrix(
        &Matrix4::from_column_slice(&state.O_T_EE)
            .remove_column(3)
            .remove_row(3),
    );
    let mut robot_ee_pose_d = Isometry3::from_parts(
        Vector3::new(state.O_T_EE[12], state.O_T_EE[13], state.O_T_EE[14]).into(),
        rot.into(),
    );
    let result: Result<(), franka::exception::FrankaException> = robot.control_torques(
        |state: &RobotState, _step: &Duration| -> Torques {
            match rx.try_recv() {
                Ok(delta) => {
                    robot_ee_pose_d *= delta;
                },
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Torques::new([0., 0., 0., 0., 0., 0., 0.]).motion_finished();
                },
                _=>{}
            };

            let rot = nalgebra::Rotation3::<f64>::from_matrix(
                &Matrix4::from_column_slice(&state.O_T_EE)
                    .remove_column(3)
                    .remove_row(3),
            );
            let robot_ee_pose = Isometry3::from_parts(
                Vector3::new(state.O_T_EE[12], state.O_T_EE[13], state.O_T_EE[14]).into(),
                rot.into(),
            );

            let position = robot_ee_pose.translation.vector;
            let mut orientation = *robot_ee_pose.rotation.quaternion();

            let position_d = robot_ee_pose_d.translation.vector;
            let orientation_d: UnitQuaternion<f64> = robot_ee_pose_d.rotation;

            let coriolis: SMatrix<f64, 7, 1> = model.coriolis_from_state(&state).into();
            let jacobian_array = model.zero_jacobian_from_state(&Frame::EndEffector, &state);
            let jacobian = SMatrix::<f64, 6, 7>::from_column_slice(&jacobian_array);
            // let _q = Vector7::from_column_slice(&state.q);
            let dq = SMatrix::<f64, 7, 1>::from_column_slice(&state.dq);

            let mut error: Matrix6x1<f64> = Matrix6x1::<f64>::zeros();
            {
                let mut error_head = error.fixed_view_mut::<3, 1>(0, 0);
                error_head.set_column(0, &(position - position_d));
            }

            if orientation_d.coords.dot(&orientation.coords) < 0. {
                orientation.coords = -orientation.coords;
            }
            let orientation = UnitQuaternion::new_normalize(orientation);
            let error_quaternion: UnitQuaternion<f64> = orientation.inverse() * orientation_d;
            {
                let mut error_tail = error.fixed_view_mut::<3, 1>(3, 0);
                error_tail.copy_from(
                    &-(robot_ee_pose.rotation.to_rotation_matrix()
                        * Vector3::new(error_quaternion.i, error_quaternion.j, error_quaternion.k)),
                );
            }
            let tau_task = jacobian.transpose() * (-stiffness * error - damping * (jacobian * dq));
            let tau_d = tau_task + coriolis;
            Torques::new([
                tau_d[0], tau_d[1], tau_d[2], tau_d[3], tau_d[4], tau_d[5], tau_d[6],
            ])
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
