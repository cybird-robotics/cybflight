// NMPCDroneController.cpp - UE5 wrapper implementation for NMPC quadrotor controller
// Copyright 2024 FSC Lab - MIT License

#include "NMPCDroneController.h"

FNMPCDroneController::FNMPCDroneController()
{
}

void FNMPCDroneController::Initialize(float Mass,
                                        float Gravity,
                                        float MinThrust,
                                        float MaxThrust,
                                        const FVector& MaxAngularRates,
                                        int HorizonLength,
                                        float PredictionDt,
                                        const FVector& WeightPosition,
                                        const FVector& WeightAttitude,
                                        const FVector& WeightVelocity,
                                        const FVector4& WeightControl)
{
    DroneMass = Mass;
    GravityAcc = Gravity;

    // Convert UE5 FVector weights to NMPC Vector3
    NMPC::Vector3 WeightPos{WeightPosition.X, WeightPosition.Y, WeightPosition.Z};
    NMPC::Vector3 WeightAtt{WeightAttitude.X, WeightAttitude.Y, WeightAttitude.Z};
    NMPC::Vector3 WeightVel{WeightVelocity.X, WeightVelocity.Y, WeightVelocity.Z};
    NMPC::NMPCModel::VXU WeightU;
    WeightU << WeightControl.X, WeightControl.Y, WeightControl.Z, WeightControl.W;

    // Initialize model
    Model.Initialize(HorizonLength, PredictionDt, Mass, Gravity, WeightPos, WeightAtt, WeightVel, WeightU);

    // Set control constraints
    std::vector<int> u_idx;
    std::vector<NMPC::Scalar> u_lb;
    std::vector<NMPC::Scalar> u_ub;

    // Thrust constraint
    u_idx.push_back(0);
    u_lb.push_back(MinThrust);
    u_ub.push_back(MaxThrust);

    // Angular rate constraints per axis (Roll, Pitch, Yaw)
    // Index 1: Roll rate (omega_x)
    u_idx.push_back(1);
    u_lb.push_back(-MaxAngularRates.X);
    u_ub.push_back(MaxAngularRates.X);

    // Index 2: Pitch rate (omega_y)
    u_idx.push_back(2);
    u_lb.push_back(-MaxAngularRates.Y);
    u_ub.push_back(MaxAngularRates.Y);

    // Index 3: Yaw rate (omega_z)
    u_idx.push_back(3);
    u_lb.push_back(-MaxAngularRates.Z);
    u_ub.push_back(MaxAngularRates.Z);

    Model.SetUConstraints(u_idx, u_lb, u_ub);

    // Set up solver
    Solver.SetModel(Model);

    bIsInitialized = true;

    UE_LOG(LogTemp, Log, TEXT("NMPCDroneController initialized: Mass=%.2f, Horizon=%d, dt=%.3f, MaxRates=(%.2f, %.2f, %.2f) rad/s"),
           Mass, HorizonLength, PredictionDt, MaxAngularRates.X, MaxAngularRates.Y, MaxAngularRates.Z);
}

void FNMPCDroneController::SetWeights(const FVector& PositionWeight,
                                       const FVector& AttitudeWeight,
                                       const FVector& VelocityWeight,
                                       const FVector4& ControlWeight)
{
    if (!bIsInitialized)
    {
        UE_LOG(LogTemp, Warning, TEXT("NMPCDroneController::SetWeights called before Initialize!"));
        return;
    }

    Model.weightPos_ << PositionWeight.X, PositionWeight.Y, PositionWeight.Z;
    Model.weightAtt_ << AttitudeWeight.X, AttitudeWeight.Y, AttitudeWeight.Z;
    Model.weightVel_ << VelocityWeight.X, VelocityWeight.Y, VelocityWeight.Z;
    Model.weightU_ << ControlWeight.X, ControlWeight.Y, ControlWeight.Z, ControlWeight.W;

    // Update solver with new model
    Solver.SetModel(Model);
}

NMPC::NMPCModel::VXX FNMPCDroneController::ConvertToNMPCState(const FVector& Position,
                                                               const FQuat& Orientation,
                                                               const FVector& Velocity) const
{
    NMPC::NMPCModel::VXX state;

    // NOTE: Position, orientation, and velocity are already converted to NMPC coordinate system
    // in QuadrotorPawn_NMPC.cpp (UEToNMPCPosition, UEToNMPCQuat, UEToNMPCVelocity)
    // So we just pass them through here without additional conversion

    // Position (meters) - already in NMPC frame
    state(0) = Position.X;
    state(1) = Position.Y;
    state(2) = Position.Z;

    // Quaternion - already in NMPC frame
    state(3) = Orientation.X;
    state(4) = Orientation.Y;
    state(5) = Orientation.Z;
    state(6) = Orientation.W;

    // Velocity (m/s) - already in NMPC frame
    state(7) = Velocity.X;
    state(8) = Velocity.Y;
    state(9) = Velocity.Z;

    return state;
}

void FNMPCDroneController::ConvertFromNMPCControl(const NMPC::NMPCModel::VXU& Control)
{
    // Extract collective thrust (already in Newtons)
    ComputedThrust = static_cast<float>(Control(0));

    // Extract angular velocity and convert from rad/s to deg/s
    // Also convert from robotics right-handed back to UE5 left-handed
    // For Y-axis mirror: negate X (roll) and Z (yaw), keep Y (pitch) same
    // ComputedAngularVelocity.X = static_cast<float>(Control(1)) * NMPC::kDegPerRad;  // Negate roll
    // ComputedAngularVelocity.Y = static_cast<float>(Control(2)) * NMPC::kDegPerRad;   // Pitch unchanged
    // ComputedAngularVelocity.Z = static_cast<float>(Control(3)) * NMPC::kDegPerRad;  // Negate yaw

    ComputedAngularVelocity.X = static_cast<float>(Control(1));  // Negate roll
    ComputedAngularVelocity.Y = static_cast<float>(Control(2));   // Pitch unchanged
    ComputedAngularVelocity.Z = static_cast<float>(Control(3));  // Negate yaw
}

bool FNMPCDroneController::ComputeControl(const FVector& CurrentPosition,
                                           const FQuat& CurrentOrientation,
                                           const FVector& CurrentVelocity,
                                           const FVector& CurrentAngularVelocity,
                                           const FVector& TargetPosition,
                                           const FQuat& TargetOrientation,
                                           const FVector& TargetVelocity)
{
    if (!bIsInitialized)
    {
        UE_LOG(LogTemp, Warning, TEXT("NMPCDroneController::ComputeControl called before Initialize!"));
        return false;
    }

    // Convert current state
    NMPC::NMPCModel::VXX x_init = ConvertToNMPCState(CurrentPosition, CurrentOrientation, CurrentVelocity);

    // Build reference trajectory (all points set to target for now)
    std::vector<NMPC::NMPCModel::VXX> x_refs;
    std::vector<NMPC::NMPCModel::VXU> u_refs;

    NMPC::NMPCModel::VXX x_ref = ConvertToNMPCState(TargetPosition, TargetOrientation, TargetVelocity);

    // Default control reference (hover)
    NMPC::NMPCModel::VXU u_ref;
    u_ref << DroneMass * GravityAcc, 0.0, 0.0, 0.0;

    // UE_LOG(LogTemp, Log, TEXT("NMPC Ref: HoverThrust=%.2f, Mass=%.2f, Gravity=%.2f"),
    //     DroneMass * GravityAcc, DroneMass, GravityAcc);

    // Debug: log the state being passed to NMPC
    UE_LOG(LogTemp, Log, TEXT("NMPC State: pos=(%.2f,%.2f,%.2f) quat=(%.3f,%.3f,%.3f,%.3f) vel=(%.2f,%.2f,%.2f)"),
        x_init(0), x_init(1), x_init(2),
        x_init(3), x_init(4), x_init(5), x_init(6),
        x_init(7), x_init(8), x_init(9));
    UE_LOG(LogTemp, Log, TEXT("NMPC Target: pos=(%.2f,%.2f,%.2f) quat=(%.3f,%.3f,%.3f,%.3f)"),
        x_ref(0), x_ref(1), x_ref(2),
        x_ref(3), x_ref(4), x_ref(5), x_ref(6));

    const int N = Model.N();
    x_refs.reserve(N + 1);
    u_refs.reserve(N);

    for (int i = 0; i <= N; ++i)
    {
        x_refs.push_back(x_ref);
        if (i < N)
        {
            u_refs.push_back(u_ref);
        }
    }

    Solver.GetModel().SetReferences(x_refs, u_refs);

    // Solve NMPC
    double StartTime = FPlatformTime::Seconds();

    std::vector<NMPC::NMPCModel::VXX> states;
    std::vector<NMPC::NMPCModel::VXU> controls;
    NMPC::Scalar result = Solver.Solve(x_init, 1.0e-5, states, controls);

    double EndTime = FPlatformTime::Seconds();
    LastSolveTimeMs = static_cast<float>((EndTime - StartTime) * 1000.0);

    if (std::isinf(result))
    {
        UE_LOG(LogTemp, Warning, TEXT("NMPC optimization failed!"));
        // Return hover command on failure
        // ComputedThrust = DroneMass * GravityAcc;
        // ComputedAngularVelocity = FVector::ZeroVector;
        return false;
    }

    // Extract first control command
    if (!controls.empty())
    {
        const auto& u = controls.front();
        UE_LOG(LogTemp, Log, TEXT("NMPC Raw Control: Thrust=%.3f, wx=%.4f, wy=%.4f, wz=%.4f rad/s"),
            u(0), u(1), u(2), u(3));
        ConvertFromNMPCControl(u);

        // UE_LOG(LogTemp, Log, TEXT("NMPC Converted Control: Thrust=%.3f, wx=%.4f, wy=%.4f, wz=%.4f rad/s"),
        //     u(0), ComputedAngularVelocity.X, ComputedAngularVelocity.Y, ComputedAngularVelocity.Z);
    }

    return true;
}

bool FNMPCDroneController::ComputeControlToPosition(const FVector& CurrentPosition,
                                                      const FQuat& CurrentOrientation,
                                                      const FVector& CurrentVelocity,
                                                      const FVector& CurrentAngularVelocity,
                                                      const FVector& TargetPosition)
{
    // Use identity quaternion (upright) and zero velocity as target
    return ComputeControl(CurrentPosition, CurrentOrientation, CurrentVelocity,
                          CurrentAngularVelocity, TargetPosition, FQuat::Identity, FVector::ZeroVector);
}

FVector FNMPCDroneController::GetTargetAngularVelocityRad() const
{
    // return FVector(
    //     ComputedAngularVelocity.X * NMPC::kRadPerDeg,
    //     ComputedAngularVelocity.Y * NMPC::kRadPerDeg,
    //     ComputedAngularVelocity.Z * NMPC::kRadPerDeg
    // );
    return ComputedAngularVelocity;
}

int FNMPCDroneController::GetLastSolverIterations() const
{
    return Solver.LastIterations();
}
