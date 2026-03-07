// NMPCDroneController.h - UE5 wrapper for NMPC quadrotor controller
// Provides target angular velocity and collective thrust for drone control
// Copyright 2024 FSC Lab - MIT License

#pragma once

#include "CoreMinimal.h"
#include "NMPCTypes.h"
#include "NMPCModel.h"
#include "NMPCSolver.h"

/**
 * FNMPCDroneController
 *
 * A wrapper class that provides NMPC-based control for quadrotor drones in UE5.
 *
 * Usage:
 *   1. Create an instance and call Initialize() with drone parameters
 *   2. Each tick, call ComputeControl() with current state and target state
 *   3. Get the computed thrust and angular velocity for your drone
 *
 * The controller outputs:
 *   - CollectiveThrust: Total thrust force (Newtons)
 *   - TargetAngularVelocity: Desired body angular rates (rad/s)
 */

class DRONESVSPROPS_API FNMPCDroneController
{
public:
    FNMPCDroneController();
    ~FNMPCDroneController() = default;

    //=========================================================================
    // Initialization
    //=========================================================================

    /**
     * Initialize the NMPC controller with drone parameters
     *
     * @param Mass - Drone mass in kg
     * @param Gravity - Gravity acceleration (default 9.81 m/s^2)
     * @param MinThrust - Minimum collective thrust (N)
     * @param MaxThrust - Maximum collective thrust (N)
     * @param MaxAngularRates - Maximum body angular rates per axis (rad/s) - X=Roll, Y=Pitch, Z=Yaw
     * @param HorizonLength - NMPC prediction horizon (default 20)
     * @param PredictionDt - Time step for prediction (default 0.05s)
     * @param WeightPosition - Weight for position tracking (x, y, z) - default (50, 50, 100)
     * @param WeightAttitude - Weight for attitude tracking (roll, pitch, yaw) - default (5, 5, 200)
     * @param WeightVelocity - Weight for velocity tracking - default (1, 1, 1)
     * @param WeightControl - Weight for control effort (thrust, wx, wy, wz) - default (1, 1, 1, 1)
     */
    void Initialize(float Mass,
                    float Gravity = 9.81f,
                    float MinThrust = 0.5f,
                    float MaxThrust = 30.0f,
                    const FVector& MaxAngularRates = FVector(10.0f, 10.0f, 10.0f),
                    int HorizonLength = 20,
                    float PredictionDt = 0.05f,
                    const FVector& WeightPosition = FVector(50.0, 50.0, 100.0),
                    const FVector& WeightAttitude = FVector(5.0, 5.0, 200.0),
                    const FVector& WeightVelocity = FVector(1.0, 1.0, 1.0),
                    const FVector4& WeightControl = FVector4(1.0, 1.0, 1.0, 1.0));

    /**
     * Set cost weights for the NMPC optimization
     *
     * @param PositionWeight - Weight for position tracking (x, y, z)
     * @param AttitudeWeight - Weight for attitude tracking (roll, pitch, yaw)
     * @param VelocityWeight - Weight for velocity tracking
     * @param ControlWeight - Weight for control effort
     */
    void SetWeights(const FVector& PositionWeight = FVector(50.0, 50.0, 100.0),
                    const FVector& AttitudeWeight = FVector(5.0, 5.0, 200.0),
                    const FVector& VelocityWeight = FVector(1.0, 1.0, 1.0),
                    const FVector4& ControlWeight = FVector4(1.0, 1.0, 1.0, 1.0));

    //=========================================================================
    // Control Computation
    //=========================================================================

    /**
     * Compute NMPC control command
     *
     * @param CurrentPosition - Current position in world frame (meters)
     * @param CurrentOrientation - Current orientation quaternion
     * @param CurrentVelocity - Current velocity in world frame (m/s)
     * @param CurrentAngularVelocity - Current angular velocity (rad/s)
     * @param TargetPosition - Desired position in world frame (meters)
     * @param TargetOrientation - Desired orientation quaternion
     * @param TargetVelocity - Desired velocity in world frame (m/s)
     * @return true if optimization succeeded
     */
    bool ComputeControl(const FVector& CurrentPosition,
                        const FQuat& CurrentOrientation,
                        const FVector& CurrentVelocity,
                        const FVector& CurrentAngularVelocity,
                        const FVector& TargetPosition,
                        const FQuat& TargetOrientation = FQuat::Identity,
                        const FVector& TargetVelocity = FVector::ZeroVector);

    /**
     * Simplified version: just track a target position
     */
    bool ComputeControlToPosition(const FVector& CurrentPosition,
                                   const FQuat& CurrentOrientation,
                                   const FVector& CurrentVelocity,
                                   const FVector& CurrentAngularVelocity,
                                   const FVector& TargetPosition);

    //=========================================================================
    // Output Getters
    //=========================================================================

    /** Get the computed collective thrust (Newtons) */
    float GetCollectiveThrust() const { return ComputedThrust; }

    /** Get the computed target angular velocity in body frame (deg/s - UE5 units) */
    FVector GetTargetAngularVelocity() const { return ComputedAngularVelocity; }

    /** Get the computed target angular velocity in body frame (rad/s) */
    FVector GetTargetAngularVelocityRad() const;

    /** Check if controller is initialized */
    bool IsInitialized() const { return bIsInitialized; }

    /** Get last solver iterations count */
    int GetLastSolverIterations() const;

    /** Get last solve time in milliseconds */
    float GetLastSolveTimeMs() const { return LastSolveTimeMs; }

    //=========================================================================
    // Rate Controller (for converting body rates to motor commands)
    //=========================================================================

    /**
     * Set rate controller gains (P-controller for body rate tracking)
     * Used if you want to get motor thrusts instead of just body rates
     */
    void SetRateControllerGains(const FVector& Kp) { RateControllerKp = Kp; }

private:
    // NMPC components
    NMPC::NMPCModel Model;
    NMPC::NMPCSolver<NMPC::NMPCModel> Solver;

    // State
    bool bIsInitialized = false;

    // Parameters
    float DroneMass = 1.0f;
    float GravityAcc = 9.81f;

    // Computed outputs
    float ComputedThrust = 0.0f;
    FVector ComputedAngularVelocity = FVector::ZeroVector;  // deg/s

    // Rate controller
    FVector RateControllerKp = FVector(20.0f, 20.0f, 8.0f);

    // Performance tracking
    float LastSolveTimeMs = 0.0f;

    // Helper functions
    NMPC::NMPCModel::VXX ConvertToNMPCState(const FVector& Position,
                                             const FQuat& Orientation,
                                             const FVector& Velocity) const;

    void ConvertFromNMPCControl(const NMPC::NMPCModel::VXU& Control);
};
