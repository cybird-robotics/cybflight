// AFPVDronePawn.cpp
#include "AFPVDronePawn.h"
#include "Components/CapsuleComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "Kismet/KismetMathLibrary.h"
#include "DrawDebugHelpers.h"

AAFPVDronePawn::AAFPVDronePawn()
{
    PrimaryActorTick.bCanEverTick = true;

    // Root capsule for collision
    CapsuleComponent = CreateDefaultSubobject<UCapsuleComponent>(TEXT("CapsuleComponent"));
    CapsuleComponent->InitCapsuleSize(50.0f, 50.0f);
    CapsuleComponent->SetCollisionProfileName(TEXT("Pawn"));
    RootComponent = CapsuleComponent;

    // Drone mesh
    DroneMesh = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("DroneMesh"));
    DroneMesh->SetupAttachment(RootComponent);
    DroneMesh->SetRelativeLocation(FVector(0.0f, 0.0f, 0.0f));
    DroneMesh->SetRelativeRotation(FRotator(0.0f, -90.0f, 0.0f));
    DroneMesh->SetOwnerNoSee(true);  // Hide from own camera (first-person view)

    // FPV Camera - attached directly to root with offset
    Camera = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
    Camera->SetupAttachment(RootComponent);
    Camera->SetRelativeLocation(CameraOffset);
    Camera->SetFieldOfView(CameraFOV);

    // Floating pawn movement with collision
    MovementComponent = CreateDefaultSubobject<UFloatingPawnMovement>(TEXT("MovementComponent"));
    MovementComponent->MaxSpeed = MaxHorizontalSpeed;
    MovementComponent->Acceleration = 2000.0f;
    MovementComponent->Deceleration = 2000.0f;
    MovementComponent->TurningBoost = 8.0f;

    // ACRO mode: we control rotation directly, no controller rotation
    bUseControllerRotationYaw = false;
    bUseControllerRotationPitch = false;
    bUseControllerRotationRoll = false;

    AutoPossessPlayer = EAutoReceiveInput::Player0;
}

// Called when the game starts or when spawned
void AAFPVDronePawn::BeginPlay()
{
    Super::BeginPlay();

    // Initialize camera tilt to default
    CurrentCameraTilt = DefaultCameraTiltAngle;
    UpdateCameraTilt();

    // Bind collision event
    if (CapsuleComponent)
    {
        CapsuleComponent->OnComponentHit.AddDynamic(this, &AAFPVDronePawn::OnHit);
    }

    // Setup Enhanced Input
    if (APlayerController* PlayerController = Cast<APlayerController>(GetController()))
    {
        if (UEnhancedInputLocalPlayerSubsystem* Subsystem =
            ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(PlayerController->GetLocalPlayer()))
        {
            if (InputMappingContext)
            {
                Subsystem->AddMappingContext(InputMappingContext, 0);
                UE_LOG(LogTemp, Log, TEXT("AAFPVDronePawn: Input Mapping Context added"));
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("AAFPVDronePawn: InputMappingContext is null! Please assign in Blueprint."));
            }
        }
    }

    // Initialize NMPC Controller if enabled
    if (bEnableNMPCControl)
    {
        // Initialize NMPC physics state
        NMPCPosition = GetActorLocation();
        NMPCVelocity = FVector::ZeroVector;

        // Initialize camera rotation state for altitude hold mode
        if (APlayerController* PC = Cast<APlayerController>(GetController()))
        {
            TargetControlRotation = PC->GetControlRotation();
            CurrentControlRotation = TargetControlRotation;
        }

        InitializeNMPCController();
    }
}

// Called every frame
void AAFPVDronePawn::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // NMPC Control Mode
    // if (bEnableNMPCControl && NMPCController.IsValid())
    if (bEnableNMPCControl)
    {
        // Speed boost logic: linearly increase velocity while boost is active
        if (bSpeedBoostActive && NMPCControlMode == ENMPCControlMode::AltitudeHold)
        {
            float OldVelocity = NMPCAltitudeHoldMaxVelocity;
            NMPCAltitudeHoldMaxVelocity += SpeedBoostAcceleration * DeltaTime;
            NMPCAltitudeHoldMaxVelocity = FMath::Clamp(NMPCAltitudeHoldMaxVelocity, BaseMaxVelocity, SpeedBoostMaxVelocity);

            // Log every second (approximately)
            static float LogTimer = 0.0f;
            LogTimer += DeltaTime;
            if (LogTimer >= 1.0f)
            {
                UE_LOG(LogTemp, Warning, TEXT("Speed Boost ACTIVE - Velocity: %.0f -> %.0f cm/s (Max: %.0f)"),
                    OldVelocity, NMPCAltitudeHoldMaxVelocity, SpeedBoostMaxVelocity);
                LogTimer = 0.0f;
            }
        }

        // Update target position from joystick input
        UpdateNMPCTargetFromInput(DeltaTime);


        UpdateNMPCControl(DeltaTime);

        int32 NumSubsteps = FMath::Max(1, FMath::CeilToInt(DeltaTime / PhysicsSubstepTime));
        float SubstepDT = DeltaTime / NumSubsteps;
        // Run physics simulation with multiple substeps
        for (int32 i = 0; i < NumSubsteps; i++)
        {
            ApplyNMPCControl(SubstepDT);
        }

        if (NMPCControlMode == ENMPCControlMode::AltitudeHold && Camera)
        {
            // Get drone's world rotation
            FRotator DroneRotation = GetActorRotation();

            // Camera should follow drone's yaw, but use independent pitch
            FRotator DesiredWorldCameraRotation;
            DesiredWorldCameraRotation.Yaw = CurrentControlRotation.Yaw;  // Follow drone yaw
            // DesiredWorldCameraRotation.Pitch = CurrentControlRotation.Pitch;  // Independent pitch
            DesiredWorldCameraRotation.Pitch = CurrentControlRotation.Pitch;  // Independent pitch
            DesiredWorldCameraRotation.Roll = 0.0f;  // No roll

            // Calculate relative rotation: what rotation does camera need relative to drone?
            FQuat DroneQuat = DroneRotation.Quaternion();
            FQuat DesiredWorldCameraQuat = DesiredWorldCameraRotation.Quaternion();
            FQuat RelativeCameraQuat = DroneQuat.Inverse() * DesiredWorldCameraQuat;

            // Set camera's relative rotation
            Camera->SetRelativeRotation(RelativeCameraQuat.Rotator());
        }


        if (bShowNMPCDebugInfo)
        {
            PrintNMPCDebugInfo();
        }

        // Apply gravity and air drag
        // NMPC thrust accounts for gravity, so physics simulation needs both:
        // - Thrust (upward along drone body) from ApplyNMPCControl
        // - Gravity (downward in world frame) from ApplyGravity
        // ApplyAirDrag(DeltaTime);

        // if (bShowDebugVectors)
        // {
        //     DrawDebugVectors();
        // }

        return; // Skip ACRO mode processing
    }

    // High-frequency sub-stepped physics simulation (ACRO Mode)
    // This dramatically improves accuracy by integrating with smaller timesteps

    // Calculate number of substeps needed
    int32 NumSubsteps = FMath::Max(1, FMath::CeilToInt(DeltaTime / PhysicsSubstepTime));
    float SubstepDT = DeltaTime / NumSubsteps;

    // Run physics simulation with multiple substeps
    for (int32 i = 0; i < NumSubsteps; i++)
    {
        // Update angular velocity based on input rates
        UpdateAngularVelocity(SubstepDT);

        // Apply rotation FIRST - this updates the drone's orientation
        // Critical: we need the new orientation before calculating thrust vector
        ApplyRotation(SubstepDT);

        // Now apply forces with the UPDATED orientation
        // Throttle uses current drone up vector (body frame Z-axis)
        float ThrustMagnitude = CurrentThrottle * GravityAcceleration * ThrottlePowerMultiplier;
        ApplyThrottle(ThrustMagnitude, SubstepDT);

        // Gravity always acts in world down
        ApplyGravity(SubstepDT);

        // Ground effect if enabled
        if (bEnableGroundEffect)
        {
            ApplyGroundEffect(SubstepDT);
        }

        // Air drag
        ApplyAirDrag(SubstepDT);
    }

    // Debug visualization (only once per frame, not per substep)
    if (bShowDebugVectors)
    {
        DrawDebugVectors();
    }

    if (bShowDebugInfo)
    {
        PrintDebugInfo();
    }
}

void AAFPVDronePawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
    Super::SetupPlayerInputComponent(PlayerInputComponent);

    if (UEnhancedInputComponent* EnhancedInputComponent = Cast<UEnhancedInputComponent>(PlayerInputComponent))
    {
        // Yaw/Throttle (Left Stick 2D Axis - X is Yaw, Y is Throttle)
        if (YawThrottleAction)
        {
            EnhancedInputComponent->BindAction(YawThrottleAction, ETriggerEvent::Triggered, this, &AAFPVDronePawn::OnYawThrottle);
            EnhancedInputComponent->BindAction(YawThrottleAction, ETriggerEvent::Completed, this, &AAFPVDronePawn::OnYawThrottle);
        }

        // Roll/Pitch (Right Stick)
        if (RollPitchAction)
        {
            EnhancedInputComponent->BindAction(RollPitchAction, ETriggerEvent::Triggered, this, &AAFPVDronePawn::OnRollPitch);
            EnhancedInputComponent->BindAction(RollPitchAction, ETriggerEvent::Completed, this, &AAFPVDronePawn::OnRollPitch);
        }

        // Camera Tilt (D-Pad Up/Down)
        if (CameraTiltAction)
        {
            EnhancedInputComponent->BindAction(CameraTiltAction, ETriggerEvent::Triggered, this, &AAFPVDronePawn::OnCameraTilt);
        }

        // Mouse Look (for altitude hold mode)
        if (MouseLookAction)
        {
            EnhancedInputComponent->BindAction(MouseLookAction, ETriggerEvent::Triggered, this, &AAFPVDronePawn::OnMouseLook);
        }

        // Keyboard Move (WASD for altitude hold mode)
        if (KeyboardMoveAction)
        {
            EnhancedInputComponent->BindAction(KeyboardMoveAction, ETriggerEvent::Triggered, this, &AAFPVDronePawn::OnKeyboardMove);
            EnhancedInputComponent->BindAction(KeyboardMoveAction, ETriggerEvent::Completed, this, &AAFPVDronePawn::OnKeyboardMove);
        }

        // Mouse Wheel (for adjusting max velocity in altitude hold mode)
        if (MouseWheelAction)
        {
            EnhancedInputComponent->BindAction(MouseWheelAction, ETriggerEvent::Triggered, this, &AAFPVDronePawn::OnMouseWheel);
        }

        // Speed Boost (Shift key for altitude hold mode)
        if (SpeedBoostAction)
        {
            EnhancedInputComponent->BindAction(SpeedBoostAction, ETriggerEvent::Triggered, this, &AAFPVDronePawn::OnSpeedBoost);
            EnhancedInputComponent->BindAction(SpeedBoostAction, ETriggerEvent::Completed, this, &AAFPVDronePawn::OnSpeedBoost);
        }
    }
}

// ========== INPUT HANDLERS ==========

void AAFPVDronePawn::OnYawThrottle(const FInputActionValue& Value)
{
    // Get 2D stick input (Left Stick - X is Yaw, Y is Throttle)
    FVector2D StickInput = Value.Get<FVector2D>();

    // Store raw input for NMPC target control
    YawInputRaw = StickInput.X;
    ThrottleInputRaw = StickInput.Y;

    // Yaw from X axis - apply deadzone then Betaflight rates
    float DeadzoneYaw = ApplyDeadzone(StickInput.X, RateDeadzone);
    YawInput = ApplyBetaflightRates(DeadzoneYaw, YawRCRate, YawSuperRate, YawExpo);

    // Throttle from Y axis:
    // Stick down (-1) = 0 throttle
    // Stick center (0) = 0 throttle
    // Stick up (+1) = 1 throttle
    // Use FMath::Max to clamp negative values to 0
    ThrottleInput = FMath::Max(StickInput.Y, 0.0f); // Only positive values [0,1]

    // Apply throttle curve and clamp
    CurrentThrottle = FMath::Clamp(
        ApplyCurve(ThrottleInput, ThrottleResponseCurve),
        MinThrottle,
        MaxThrottle
    );

    // UE_LOG(LogTemp, Log, TEXT("YawThrottle - Yaw: %.2f, Throttle: %.2f"), YawInput, CurrentThrottle);
}

void AAFPVDronePawn::OnRollPitch(const FInputActionValue& Value)
{
    // Get 2D stick input (Right Stick)
    FVector2D StickInput = Value.Get<FVector2D>();

    // Store raw input for NMPC target control
    RollInputRaw = StickInput.X;
    PitchInputRaw = StickInput.Y;

    // Apply deadzone then Betaflight rates for Roll
    float DeadzoneRoll = ApplyDeadzone(StickInput.X, RateDeadzone);
    RollInput = ApplyBetaflightRates(DeadzoneRoll, RollRCRate, RollSuperRate, RollExpo);

    // Apply deadzone then Betaflight rates for Pitch
    // Stick Up (+Y) = Pitch Forward (positive pitch), Stick Down (-Y) = Pitch Back (negative pitch)
    float DeadzonePitch = ApplyDeadzone(StickInput.Y, RateDeadzone);
    PitchInput = ApplyBetaflightRates(DeadzonePitch, PitchRCRate, PitchSuperRate, PitchExpo);

    // UE_LOG(LogTemp, Log, TEXT("RollPitch - Roll: %.2f, Pitch: %.2f"), RollInput, PitchInput);
}

void AAFPVDronePawn::OnCameraTilt(const FInputActionValue& Value)
{
    // Get D-Pad input (Up = +1, Down = -1)
    float TiltInput = Value.Get<float>();

    // Adjust camera tilt angle
    CurrentCameraTilt += TiltInput * CameraTiltAdjustmentSpeed;
    CurrentCameraTilt = FMath::Clamp(CurrentCameraTilt, MinCameraTiltAngle, MaxCameraTiltAngle);

    UpdateCameraTilt();

    UE_LOG(LogTemp, Log, TEXT("Camera Tilt: %.1f degrees"), CurrentCameraTilt);
}

void AAFPVDronePawn::OnMouseLook(const FInputActionValue& Value)
{
    // Get mouse delta (Vector2D)
    MouseDelta = Value.Get<FVector2D>();
}

void AAFPVDronePawn::OnKeyboardMove(const FInputActionValue& Value)
{
    // Get WASD input (Vector2D: X = A/D, Y = W/S)
    KeyboardMoveInput = Value.Get<FVector2D>();
}

void AAFPVDronePawn::OnMouseWheel(const FInputActionValue& Value)
{
    // Get mouse wheel delta (float: positive = scroll up, negative = scroll down)
    float WheelDelta = Value.Get<float>();

    // Adjust max velocity in altitude hold mode
    if (NMPCControlMode == ENMPCControlMode::AltitudeHold)
    {
        NMPCAltitudeHoldMaxVelocity += WheelDelta * MouseWheelVelocityStep;

        // Clamp to reasonable range
        NMPCAltitudeHoldMaxVelocity = FMath::Clamp(NMPCAltitudeHoldMaxVelocity, 100.0f, 2000.0f);

        // If boost is active, update the base velocity too
        if (bSpeedBoostActive)
        {
            BaseMaxVelocity = NMPCAltitudeHoldMaxVelocity;
        }

        UE_LOG(LogTemp, Log, TEXT("Max Velocity adjusted: %.0f cm/s (%.1f m/s)"),
            NMPCAltitudeHoldMaxVelocity, NMPCAltitudeHoldMaxVelocity * 0.01f);
    }
}

void AAFPVDronePawn::OnSpeedBoost(const FInputActionValue& Value)
{
    // Only work in altitude hold mode
    if (NMPCControlMode != ENMPCControlMode::AltitudeHold)
    {
        return;
    }

    // Get the float value (1.0 when pressed, 0.0 when released)
    float InputValue = Value.Get<float>();
    bool bIsPressed = InputValue > 0.1f;  // Use 0.1 threshold instead of 0.5

    UE_LOG(LogTemp, Warning, TEXT("OnSpeedBoost called - InputValue: %.2f, bIsPressed: %d, bSpeedBoostActive: %d"),
        InputValue, bIsPressed, bSpeedBoostActive);

    if (bIsPressed && !bSpeedBoostActive)
    {
        // Shift pressed: Start boosting
        bSpeedBoostActive = true;
        BaseMaxVelocity = NMPCAltitudeHoldMaxVelocity;  // Store current velocity
        UE_LOG(LogTemp, Warning, TEXT(">>> Speed Boost STARTED - Base: %.0f cm/s"), BaseMaxVelocity);
    }
    else if (!bIsPressed && bSpeedBoostActive)
    {
        // Shift released: Return to base velocity
        // Only end boost if it was actually started
        bSpeedBoostActive = false;
        NMPCAltitudeHoldMaxVelocity = BaseMaxVelocity;  // Restore to velocity when boost started
        UE_LOG(LogTemp, Warning, TEXT("<<< Speed Boost ENDED - Restored to: %.0f cm/s"), BaseMaxVelocity);
    }
}

// ========== ACRO FLIGHT LOGIC ==========

void AAFPVDronePawn::UpdateAngularVelocity(float DeltaTime)
{
    // Calculate target angular velocities from stick inputs
    FVector TargetAngularVel;
    TargetAngularVel.X = RollInput * MaxRollRate;    // Roll
    TargetAngularVel.Y = PitchInput * MaxPitchRate;  // Pitch
    TargetAngularVel.Z = YawInput * MaxYawRate;      // Yaw

    // Interpolate current angular velocity towards target (adds damping)
    AngularVelocity = FMath::VInterpTo(
        AngularVelocity,
        TargetAngularVel,
        DeltaTime,
        AngularDamping
    );
}

void AAFPVDronePawn::ApplyRotation(float DeltaTime)
{
    // Convert angular velocity (deg/s) to rotation change
    FRotator DeltaRotation;
    DeltaRotation.Roll = AngularVelocity.X * DeltaTime;
    DeltaRotation.Pitch = AngularVelocity.Y * DeltaTime;
    DeltaRotation.Yaw = AngularVelocity.Z * DeltaTime;

    // Apply rotation in local space (body-rate control)
    FRotator CurrentRotation = GetActorRotation();
    FQuat CurrentQuat = CurrentRotation.Quaternion();
    FQuat DeltaQuat = DeltaRotation.Quaternion();
    FQuat NewQuat = CurrentQuat * DeltaQuat;

    SetActorRotation(NewQuat);
}

void AAFPVDronePawn::ApplyThrottle(float ThrustMagnitude, float DeltaTime)
{
    // Calculate thrust magnitude based on throttle
    // ThrottlePowerMultiplier controls how much thrust relative to gravity
    // Example: 3.0 means max throttle = 3x gravity force (can climb fast)
    // float ThrustMagnitude = CurrentThrottle * GravityAcceleration * ThrottlePowerMultiplier;

    // Apply thrust along drone's local UP vector (Z-axis)
    // This is the key: thrust is always "up" relative to the drone, not the world
    // When the drone tilts, this thrust vector tilts with it, creating movement in that direction
    FVector DroneUpVector = GetActorUpVector();
    FVector ThrustAcceleration = DroneUpVector * ThrustMagnitude;

    // Apply the thrust acceleration
    FVector ThrustVelocityChange = ThrustAcceleration * DeltaTime;
    MovementComponent->Velocity += ThrustVelocityChange;
}

void AAFPVDronePawn::ApplyGravity(float DeltaTime)
{
    // Apply gravity acceleration
    FVector GravityVector = FVector(0.0f, 0.0f, -GravityAcceleration * DeltaTime);
    MovementComponent->Velocity += GravityVector;
}

void AAFPVDronePawn::ApplyGroundEffect(float DeltaTime)
{
    // Get current altitude (Z position)
    float CurrentAltitude = GetActorLocation().Z;

    // Apply ground effect when near the ground
    if (CurrentAltitude > 0.0f && CurrentAltitude < GroundEffectHeight)
    {
        // Realistic ground effect model:
        // Ground effect increases thrust efficiency when close to ground
        // Effect is proportional to 1/h² (inverse square of height)
        // This creates a strong, non-linear cushion very close to ground

        // Normalized height (0 = on ground, 1 = at max ground effect height)
        float NormalizedHeight = CurrentAltitude / GroundEffectHeight;
        NormalizedHeight = FMath::Max(NormalizedHeight, 0.01f); // Prevent divide by zero

        // Non-linear ground effect using inverse square law
        // Effect ∝ 1/h² - very strong near ground, falls off rapidly
        float InverseSquare = 1.0f / (NormalizedHeight * NormalizedHeight);

        // Clamp to prevent infinite force at h=0
        InverseSquare = FMath::Min(InverseSquare, 100.0f); // Max 100x multiplier

        // Apply exponential decay for smoother transition
        // Combined model: inverse square with exponential falloff
        float EffectMultiplier = InverseSquare * FMath::Exp(-NormalizedHeight * 3.0f);

        // Scale by strength parameter and current throttle
        // Ground effect only matters when props are spinning (throttle > 0)
        float ThrottleFactor = FMath::Max(CurrentThrottle, 0.1f); // Minimum 10% even at zero throttle
        float UpwardForce = GroundEffectStrength * EffectMultiplier * ThrottleFactor * DeltaTime;

        // Apply force in world up direction
        FVector GroundEffectVector = FVector(0.0f, 0.0f, UpwardForce);
        MovementComponent->Velocity += GroundEffectVector;

        // Optional: Add some drag near ground (ground friction)
        FVector HorizontalVel = FVector(MovementComponent->Velocity.X, MovementComponent->Velocity.Y, 0.0f);
        float GroundDrag = (1.0f - NormalizedHeight) * 0.1f; // More drag closer to ground
        MovementComponent->Velocity -= HorizontalVel * GroundDrag * DeltaTime;
    }
}

void AAFPVDronePawn::ApplyAirDrag(float DeltaTime)
{
    // Apply air resistance proportional to velocity
    FVector CurrentVel = MovementComponent->Velocity;
    FVector DragForce = -CurrentVel * AirDrag * DeltaTime;
    MovementComponent->Velocity += DragForce;
}

void AAFPVDronePawn::ClampVelocity()
{
    // // Clamp horizontal velocity
    // FVector CurrentVel = MovementComponent->Velocity;
    // FVector HorizontalVel = FVector(CurrentVel.X, CurrentVel.Y, 0.0f);

    // if (HorizontalVel.Size() > MaxHorizontalSpeed)
    // {
    //     HorizontalVel = HorizontalVel.GetSafeNormal() * MaxHorizontalSpeed;
    //     CurrentVel.X = HorizontalVel.X;
    //     CurrentVel.Y = HorizontalVel.Y;
    // }

    // // Clamp vertical velocity
    // CurrentVel.Z = FMath::Clamp(CurrentVel.Z, -MaxVerticalSpeed, MaxVerticalSpeed);

    // MovementComponent->Velocity = CurrentVel;
}

void AAFPVDronePawn::UpdateCameraTilt()
{
    if (Camera)
    {
        // Set camera pitch relative to drone body
        FRotator CameraRotation = Camera->GetRelativeRotation();
        CameraRotation.Pitch = CurrentCameraTilt;
        Camera->SetRelativeRotation(CameraRotation);
    }
}

// ========== COLLISION ==========

void AAFPVDronePawn::OnHit(UPrimitiveComponent* HitComponent, AActor* OtherActor,
    UPrimitiveComponent* OtherComp, FVector NormalImpulse, const FHitResult& Hit)
{
    if (!bEnableCollision || !OtherActor || OtherActor == this)
    {
        return;
    }

    UE_LOG(LogTemp, Warning, TEXT("AAFPVDronePawn hit: %s at location %s"),
        *OtherActor->GetName(),
        *Hit.ImpactPoint.ToString());

    // Reduce velocity on collision (bounce effect)
    FVector CurrentVel = MovementComponent->Velocity;

    // Reflect velocity off the collision normal and apply damping
    FVector ReflectedVel = CurrentVel.MirrorByVector(Hit.ImpactNormal) * CollisionDamping;
    MovementComponent->Velocity = ReflectedVel;

    // Reduce angular velocity on impact
    AngularVelocity *= CollisionDamping;

    // if (GEngine)
    // {
    //     GEngine->AddOnScreenDebugMessage(-1, 2.0f, FColor::Red,
    //         FString::Printf(TEXT("Collision! Velocity reduced.")));
    // }
}

// ========== DEBUG ==========

void AAFPVDronePawn::DrawDebugVectors()
{
    if (!GetWorld()) return;

    FVector ActorLocation = GetActorLocation();

    // Draw velocity vector (yellow)
    FVector Velocity = MovementComponent->Velocity;
    if (!Velocity.IsNearlyZero())
    {
        FVector VelocityEnd = ActorLocation + Velocity.GetSafeNormal() * 200.0f;
        DrawDebugDirectionalArrow(
            GetWorld(),
            ActorLocation,
            VelocityEnd,
            50.0f,
            FColor::Yellow,
            false,
            -1.0f,
            0,
            3.0f
        );
    }

    // Draw forward direction (red)
    FVector ForwardEnd = ActorLocation + GetActorForwardVector() * 200.0f;
    DrawDebugDirectionalArrow(
        GetWorld(),
        ActorLocation,
        ForwardEnd,
        50.0f,
        FColor::Red,
        false,
        -1.0f,
        0,
        3.0f
    );

    // Draw up direction (blue)
    FVector UpEnd = ActorLocation + GetActorUpVector() * 150.0f;
    DrawDebugDirectionalArrow(
        GetWorld(),
        ActorLocation,
        UpEnd,
        40.0f,
        FColor::Blue,
        false,
        -1.0f,
        0,
        2.0f
    );

    // Draw right direction (green)
    FVector RightEnd = ActorLocation + GetActorRightVector() * 150.0f;
    DrawDebugDirectionalArrow(
        GetWorld(),
        ActorLocation,
        RightEnd,
        40.0f,
        FColor::Green,
        false,
        -1.0f,
        0,
        2.0f
    );

    // Draw angular velocity indicator (cyan sphere scaled by rotation rate)
    float AngularSpeed = AngularVelocity.Size();
    if (AngularSpeed > 1.0f)
    {
        DrawDebugSphere(
            GetWorld(),
            ActorLocation,
            FMath::Clamp(AngularSpeed / 10.0f, 10.0f, 50.0f),
            12,
            FColor::Cyan,
            false,
            -1.0f,
            0,
            2.0f
        );
    }
}

void AAFPVDronePawn::PrintDebugInfo()
{
    if (!GEngine) return;

    FVector Velocity = MovementComponent->Velocity;
    float VelocityMagnitude = Velocity.Size();
    FVector HorizontalVel = FVector(Velocity.X, Velocity.Y, 0.0f);
    float HorizontalSpeed = HorizontalVel.Size();
    FRotator CurrentRot = GetActorRotation();

    // Print on screen debug info
    GEngine->AddOnScreenDebugMessage(10, 0.0f, FColor::White,
        FString::Printf(TEXT("=== AAFPVDronePawn ACRO Mode ===")));

    GEngine->AddOnScreenDebugMessage(11, 0.0f, FColor::Cyan,
        FString::Printf(TEXT("Throttle: %.2f (%.0f%%)"), CurrentThrottle, CurrentThrottle * 100.0f));

    GEngine->AddOnScreenDebugMessage(12, 0.0f, FColor::Yellow,
        FString::Printf(TEXT("Velocity: %.1f (H: %.1f, V: %.1f)"),
            VelocityMagnitude, HorizontalSpeed, Velocity.Z));

    GEngine->AddOnScreenDebugMessage(13, 0.0f, FColor::White,
        FString::Printf(TEXT("Rotation: P=%.1f Y=%.1f R=%.1f"),
            CurrentRot.Pitch, CurrentRot.Yaw, CurrentRot.Roll));

    GEngine->AddOnScreenDebugMessage(14, 0.0f, FColor::Magenta,
        FString::Printf(TEXT("Angular Vel: R=%.1f P=%.1f Y=%.1f deg/s"),
            AngularVelocity.X, AngularVelocity.Y, AngularVelocity.Z));

    GEngine->AddOnScreenDebugMessage(15, 0.0f, FColor::Orange,
        FString::Printf(TEXT("Stick Input: Roll=%.2f Pitch=%.2f Yaw=%.2f"),
            RollInput, PitchInput, YawInput));

    GEngine->AddOnScreenDebugMessage(16, 0.0f, FColor::Green,
        FString::Printf(TEXT("Altitude: %.1f"), GetActorLocation().Z));

    GEngine->AddOnScreenDebugMessage(17, 0.0f, FColor::Blue,
        FString::Printf(TEXT("Camera Tilt: %.1f degrees"), CurrentCameraTilt));
}

// ========== UTILITY ==========

float AAFPVDronePawn::ApplyCurve(float Input, float Curve) const
{
    // Apply exponential curve to input
    // Curve = 1.0 is linear, > 1.0 gives more fine control at center
    float Sign = FMath::Sign(Input);
    float AbsInput = FMath::Abs(Input);
    return Sign * FMath::Pow(AbsInput, Curve);
}

float AAFPVDronePawn::ApplyDeadzone(float Input, float Deadzone) const
{
    // Apply deadzone to stick input
    if (FMath::Abs(Input) < Deadzone)
    {
        return 0.0f;
    }

    // Scale input to maintain smooth transition after deadzone
    float Sign = FMath::Sign(Input);
    float ScaledInput = (FMath::Abs(Input) - Deadzone) / (1.0f - Deadzone);
    return Sign * FMath::Clamp(ScaledInput, 0.0f, 1.0f);
}

float AAFPVDronePawn::ApplyBetaflightRates(float Input, float RCRate, float SuperRate, float Expo) const
{
    // Betaflight rate calculation (Actual Rates mode)
    // Input: stick deflection [-1, 1]
    // Returns: normalized rate output [0, 1]

    float AbsInput = FMath::Abs(Input);
    float Sign = FMath::Sign(Input);

    // Apply expo curve (reduces sensitivity at center)
    float ExpoInput = AbsInput * (1.0f - Expo) + FMath::Pow(AbsInput, 3.0f) * Expo;

    // Calculate rate with RC Rate and Super Rate
    // RC Rate: base sensitivity multiplier
    // Super Rate: increases max rate at stick ends (0.0 = linear, 1.0 = maximum curve)
    float RCFactor = RCRate * 200.0f; // Scale RC Rate to degrees/second range
    float SuperFactor = (SuperRate * AbsInput * 500.0f); // Super rate adds more at stick ends

    // Combined rate (normalized to 0-1 range for internal use)
    float Rate = (RCFactor + SuperFactor) * ExpoInput / 1000.0f; // Normalize to reasonable range

    return Sign * FMath::Clamp(Rate, 0.0f, 1.0f);
}

// ========== NMPC CONTROLLER IMPLEMENTATION ==========

void AAFPVDronePawn::InitializeNMPCController()
{
    NMPCController = MakeUnique<FNMPCDroneController>();

    // Convert gravity from cm/s^2 to m/s^2
    float GravityMS2 = GravityAcceleration * 0.01f;

    // Calculate thrust limits based on drone parameters
    float MinThrustN = DroneMass * GravityMS2 * 0.1f;  // 10% of hover thrust
    float MaxThrustN = DroneMass * GravityMS2 * ThrottlePowerMultiplier;
    
    // Max angular rates per axis in rad/s (convert from deg/s)
    // X=Roll, Y=Pitch, Z=Yaw
    FVector MaxAngularRatesRad = FVector(
        MaxRollRate * 0.0174533f,
        MaxPitchRate * 0.0174533f,
        MaxYawRate * 0.0174533f
    );

    NMPCController->Initialize(
        DroneMass,
        GravityMS2,
        MinThrustN,
        MaxThrustN,
        MaxAngularRatesRad,
        NMPCHorizonLength,
        NMPCPredictionDt
    );

    // Set initial target to current position
    NMPCTargetPosition = GetActorLocation();
    NMPCTargetRotation = GetActorRotation();

    UE_LOG(LogTemp, Log, TEXT("NMPC Controller initialized: Mass=%.2f kg, Horizon=%d, dt=%.3fs"),
           DroneMass, NMPCHorizonLength, NMPCPredictionDt);
}

void AAFPVDronePawn::UpdateNMPCControl(float DeltaTime)
{
    if (!NMPCController.IsValid() || !NMPCController->IsInitialized())
    {
        UE_LOG(LogTemp, Warning, TEXT("NMPC: Controller not initialized"));
        return;
    }

    // Get current state in UE5 units
    FVector CurrentPositionCm = NMPCPosition; // Use manual physics position
    FQuat CurrentOrientation = GetActorQuat();
    FVector CurrentVelCmS = NMPCVelocity; // Use manual physics velocity

    // Convert to SI units for NMPC
    // Position: cm -> m
    FVector CurrentPositionM = CurrentPositionCm * 0.01f;
    // Velocity: cm/s -> m/s
    FVector CurrentVelMS = CurrentVelCmS * 0.01f;
    // Angular velocity: deg/s -> rad/s
    FVector CurrentAngVelRad = AngularVelocity * 0.0174533f;
    // Target position: cm -> m
    FVector TargetPositionM = NMPCTargetPosition * 0.01f;

    // CurrentOrientation.X *= -1.0f; // Invert roll for NMPC convention
    // CurrentOrientation.Z *= -1.0f; // Invert pitch for NMPC convention
    // CurrentOrientation.Y *= -1.0f;
    // CurrentAngVelRad.Y *= -1.0f; // Invert pitch rate for NMPC convention

    // Determine target velocity and adjust weights based on control mode
    FVector TargetVelocityMS;
    if (NMPCControlMode == ENMPCControlMode::PositionControl)
    {
        // Position control: target velocity = 0 (hover at position)
        // Use default weights with strong position tracking
        NMPCController->SetWeights(
            FVector(50.0f, 50.0f, 100.0f),  // Position weight (X, Y, Z)
            FVector(5.0f, 5.0f, 200.0f),    // Attitude weight (Roll, Pitch, Yaw)
            FVector(1.0f, 1.0f, 1.0f),      // Velocity weight
            FVector4(1.0f, 1.0f, 1.0f, 1.0f) // Control weight
        );
        TargetVelocityMS = FVector::ZeroVector;
    }
    else // ENMPCControlMode::AltitudeHold
    {
        // Altitude hold: 6-DOF velocity control
        // Set ALL position weights to ZERO for pure velocity control
        NMPCController->SetWeights(
            FVector(0.0f, 0.0f, 0.0f),       // Position weight: ignore all position tracking
            FVector(5.0f, 5.0f, 200.0f),     // Attitude weight (Roll, Pitch, Yaw)
            FVector(50.0f, 50.0f, 50.0f),    // Velocity weight: track all 3D velocity components
            FVector4(1.0f, 1.0f, 1.0f, 1.0f) // Control weight
        );
        TargetVelocityMS = NMPCTargetVelocity * 0.01f;

        // Update position reference to current position (don't care about position, only velocity)
        TargetPositionM = CurrentPositionM;
    }

    // Compute NMPC control (all inputs in SI units: meters, m/s, rad/s)
    bool bSuccess = NMPCController->ComputeControl(
        CurrentPositionM,
        CurrentOrientation,
        CurrentVelMS,
        CurrentAngVelRad,
        TargetPositionM,
        NMPCTargetRotation.Quaternion(),
        TargetVelocityMS
    );

    // Calculate position error for logging (in cm for UE5 display)
    FVector PositionError = NMPCTargetPosition - CurrentPositionCm;
    float PositionErrorMag = PositionError.Size();

    // Get current attitude angles in degrees
    FRotator CurrentRotation = GetActorRotation();
    float CurrentRoll = CurrentRotation.Roll;
    float CurrentPitch = CurrentRotation.Pitch;
    float CurrentYaw = CurrentRotation.Yaw;

    // Target attitude
    float TargetRoll = NMPCTargetRotation.Roll;
    float TargetPitch = NMPCTargetRotation.Pitch;
    float TargetYaw = NMPCTargetRotation.Yaw;

    if (bSuccess)
    {
        CachedNMPCThrust = NMPCController->GetCollectiveThrust();
        CachedNMPCAngularVelocity = NMPCController->GetTargetAngularVelocity();

        UE_LOG(LogTemp, Log, TEXT("NMPC: Pos=(%.1f,%.1f,%.1f) Target=(%.1f,%.1f,%.1f) | RPY=(%.1f,%.1f,%.1f) TargetRPY=(%.1f,%.1f,%.1f) | Thrust=%.2fN AngVel=(%.1f,%.1f,%.1f)deg/s | Iter=%d"),
            CurrentPositionM.X, CurrentPositionM.Y, CurrentPositionM.Z,
            TargetPositionM.X, TargetPositionM.Y, TargetPositionM.Z,
            CurrentRoll, CurrentPitch, CurrentYaw,
            TargetRoll, TargetPitch, TargetYaw,
            CachedNMPCThrust,
            CachedNMPCAngularVelocity.X,
            CachedNMPCAngularVelocity.Y,
            CachedNMPCAngularVelocity.Z,
            NMPCController->GetLastSolverIterations());
    }
    else
    {
        // // Fallback to hover
        // CachedNMPCThrust = DroneMass * GravityAcceleration;  // Hover thrust
        // CachedNMPCAngularVelocity = FVector::ZeroVector;

        // UE_LOG(LogTemp, Warning, TEXT("NMPC: FAILED | Pos=(%.1f, %.1f, %.1f)m | Yaw=%.1fdeg | PosErr=%.2fm | Falling back to hover (Thrust=%.2fN)"),
        //     CurrentPositionM.X, CurrentPositionM.Y, CurrentPositionM.Z,
        //     CurrentYaw,
        //     PositionErrorMag * 0.01f,
        //     CachedNMPCThrust);
    }
}

void AAFPVDronePawn::ApplyNMPCControl(float DeltaTime)
{
    // Manual Euler integration for NMPC control (bypassing FloatingPawnMovement)
    // This implements: x' = v, v' = a, q' = 0.5 * q * ω

    // 1. Update rotation using angular velocity (Euler method)
    // AngularVelocity = CachedNMPCAngularVelocity;

    // Convert angular velocity to rotation change (degrees/s -> rotation)
    FRotator DeltaRotation;
    DeltaRotation.Roll = CachedNMPCAngularVelocity.X * DeltaTime;
    DeltaRotation.Pitch = CachedNMPCAngularVelocity.Y * DeltaTime;
    DeltaRotation.Yaw = CachedNMPCAngularVelocity.Z * DeltaTime;

    // Apply rotation in local space (body-rate control)
    FRotator CurrentRotation = GetActorRotation();
    FQuat CurrentQuat = CurrentRotation.Quaternion();
    FQuat DeltaQuat = DeltaRotation.Quaternion();
    FQuat NewQuat = CurrentQuat * DeltaQuat;

    SetActorRotation(NewQuat);

    // 2. Calculate total acceleration (thrust + gravity)
    // Convert NMPC thrust (Newtons) to acceleration (cm/s^2)
    float ThrustMagnitude = (CachedNMPCThrust / DroneMass) * 100.0f;  // N/kg -> cm/s^2
    FVector DroneUpVector = GetActorUpVector();
    FVector ThrustAcceleration = DroneUpVector * ThrustMagnitude;

    // Gravity acceleration (world frame, down)
    FVector GravityVector = FVector(0.0f, 0.0f, -GravityAcceleration);

    // Total acceleration
    FVector TotalAcceleration = ThrustAcceleration + GravityVector;

    // 3. Update velocity using Euler method: v(t+dt) = v(t) + a*dt
    NMPCVelocity += TotalAcceleration * DeltaTime;

    // 4. Update position using Euler method: x(t+dt) = x(t) + v*dt
    NMPCPosition += NMPCVelocity * DeltaTime;

    // 5. Apply the computed position to the actor
    SetActorLocation(NMPCPosition);

    // Debug logging
    // UE_LOG(LogTemp, Log, TEXT("NMPC Euler: Pos=(%.1f,%.1f,%.1f) Vel=(%.1f,%.1f,%.1f) Acc=(%.1f,%.1f,%.1f)"),
    //     NMPCPosition.X, NMPCPosition.Y, NMPCPosition.Z,
    //     NMPCVelocity.X, NMPCVelocity.Y, NMPCVelocity.Z,
    //     TotalAcceleration.X, TotalAcceleration.Y, TotalAcceleration.Z);
}

void AAFPVDronePawn::UpdateNMPCTargetFromInput(float DeltaTime)
{
    // Apply deadzone to raw inputs (use member variables set in input handlers)
    float YawDz = ApplyDeadzone(YawInputRaw, RateDeadzone);
    float ThrottleDz = ApplyDeadzone(ThrottleInputRaw, RateDeadzone);
    float RollDz = ApplyDeadzone(RollInputRaw, RateDeadzone);
    float PitchDz = -ApplyDeadzone(PitchInputRaw, RateDeadzone);

    // Merge keyboard input with gamepad (WASD replaces right stick in altitude hold mode)
    // Keyboard input takes priority if non-zero
    if (FMath::Abs(KeyboardMoveInput.X) > 0.01f || FMath::Abs(KeyboardMoveInput.Y) > 0.01f)
    {
        RollDz = KeyboardMoveInput.X;   // A/D for left/right
        PitchDz = KeyboardMoveInput.Y;  // W/S for forward/backward
    }

    // Merge mouse input with gamepad (Mouse replaces left stick in altitude hold mode)
    // Mouse delta is directly used (accumulated over frame)
    float MouseYawDelta = MouseDelta.X * MouseYawSensitivity;
    float MousePitchDelta = MouseDelta.Y * MousePitchSensitivity;

    // Reset mouse delta after reading (it's per-frame delta)
    MouseDelta = FVector2D::ZeroVector;

    if (NMPCControlMode == ENMPCControlMode::PositionControl)
    {
        // ===== POSITION CONTROL MODE =====
        // Left Stick: X = Target Yaw rotation, Y = Target altitude (Z)
        // Right Stick: X = Target left/right, Y = Target forward/back

        // Update target yaw (left stick X)
        if (FMath::Abs(YawDz) > 0.0f)
        {
            NMPCTargetRotation.Yaw += YawDz * NMPCTargetYawSpeed * DeltaTime;
            NMPCTargetRotation.Yaw = FMath::UnwindDegrees(NMPCTargetRotation.Yaw);
        }

        // Update target altitude (left stick Y - throttle)
        if (FMath::Abs(ThrottleDz) > 0.0f)
        {
            NMPCTargetPosition.Z += ThrottleDz * NMPCTargetMoveSpeed * DeltaTime;
            NMPCTargetPosition.Z = FMath::Max(NMPCTargetPosition.Z, 50.0f);  // Minimum 50cm altitude
        }

        // Update target XY position relative to target yaw orientation
        if (FMath::Abs(PitchDz) > 0.0f || FMath::Abs(RollDz) > 0.0f)
        {
            // Get target yaw rotation
            float TargetYawRad = FMath::DegreesToRadians(NMPCTargetRotation.Yaw);

            // Forward/backward (pitch input moves along target's forward direction)
            float ForwardDelta = PitchDz * NMPCTargetMoveSpeed * DeltaTime;

            // Left/right (roll input moves along target's right direction)
            float RightDelta = RollDz * NMPCTargetMoveSpeed * DeltaTime;

            // Transform to world coordinates based on target yaw
            float CosYaw = FMath::Cos(TargetYawRad);
            float SinYaw = FMath::Sin(TargetYawRad);

            NMPCTargetPosition.X += ForwardDelta * CosYaw - RightDelta * SinYaw;
            NMPCTargetPosition.Y += ForwardDelta * SinYaw + RightDelta * CosYaw;
        }
    }
    else if (NMPCControlMode == ENMPCControlMode::AltitudeHold)
    {
        // ===== ALTITUDE HOLD MODE - 6-DOF CAMERA-ORIENTED VELOCITY CONTROL =====
        // Left Stick OR Mouse: Camera rotation (yaw and pitch)
        //   - Left Stick Y: Camera pitch (look up/down)
        //   - Left Stick X: Camera yaw (look left/right)
        //   - Mouse: Camera yaw (X) and pitch (Y) - takes priority over joystick
        // Right Stick OR WASD: 3D velocity direction following camera forward vector
        //   - Right Stick / WASD: forward/backward and strafe left/right

        // Update target camera rotation
        // Mouse takes priority - if mouse has input, use it; otherwise use joystick
        if (FMath::Abs(MouseYawDelta) > 0.01f || FMath::Abs(MousePitchDelta) > 0.01f)
        {
            // Mouse input (direct delta, not scaled by DeltaTime)
            TargetControlRotation.Yaw += MouseYawDelta;
            TargetControlRotation.Pitch += MousePitchDelta;
        }
        else
        {
            // Joystick input (scaled by DeltaTime for smooth rotation speed)
            TargetControlRotation.Yaw += YawDz * NMPCAltitudeHoldCameraYawSpeed * DeltaTime;
            TargetControlRotation.Pitch += ThrottleDz * NMPCAltitudeHoldCameraPitchSpeed * DeltaTime;
        }

        // Clamp pitch to prevent over-rotation
        TargetControlRotation.Pitch = FMath::Clamp(
            TargetControlRotation.Pitch,
            NMPCAltitudeHoldMinCameraPitch,
            NMPCAltitudeHoldMaxCameraPitch
        );

        // Smooth interpolation of camera rotation (like Bird implementation)
        CurrentControlRotation = TargetControlRotation;
        NMPCTargetRotation.Yaw = CurrentControlRotation.Yaw;
        NMPCTargetRotation.Pitch = 0.0;
        NMPCTargetRotation.Roll = 0.0;

        // Apply smoothed rotation to controller
        // if (APlayerController* PC = Cast<APlayerController>(GetController()))
        // {
        //     PC->SetControlRotation(CurrentControlRotation);
        // }

        // Calculate velocity based on camera orientation
        // Right Stick Y: Forward/backward in the direction camera is looking (includes vertical component)
        // Right Stick X: Side-slip (ALWAYS horizontal, never affects Z velocity)

        // Get camera forward direction (full 3D vector including pitch)
        FRotationMatrix CameraRotationMatrix(CurrentControlRotation);
        FVector CameraForward3D = CameraRotationMatrix.GetUnitAxis(EAxis::X);

        // Get horizontal right direction (based on camera yaw only, no pitch)
        FRotator HorizontalYaw(0.0f, CurrentControlRotation.Yaw, 0.0f);
        FRotationMatrix HorizontalRotationMatrix(HorizontalYaw);
        FVector HorizontalRight = HorizontalRotationMatrix.GetUnitAxis(EAxis::Y);

        // Forward velocity: Uses full 3D camera forward direction (pitch affects vertical motion)
        FVector ForwardVelocity = CameraForward3D * PitchDz * NMPCAltitudeHoldMaxVelocity;

        // Side-slip velocity: ALWAYS horizontal (right stick X)
        FVector SideVelocity = HorizontalRight * RollDz * NMPCAltitudeHoldMaxVelocity;

        // Combine: Forward (3D) + Side-slip (horizontal only)
        NMPCTargetVelocity = ForwardVelocity + SideVelocity;

        // Update position reference to current position (no position tracking, only velocity)
        NMPCTargetPosition = NMPCPosition;
    }
}

void AAFPVDronePawn::PrintNMPCDebugInfo()
{
    if (!GEngine || !NMPCController.IsValid())
    {
        return;
    }

    FVector PositionError = NMPCTargetPosition - GetActorLocation();

    GEngine->AddOnScreenDebugMessage(50, 0.0f, FColor::Cyan,
        TEXT("=== NMPC Controller ==="));

    // Show control mode
    FString ControlModeStr = (NMPCControlMode == ENMPCControlMode::PositionControl) ? TEXT("Position Control") : TEXT("Altitude Hold");
    GEngine->AddOnScreenDebugMessage(51, 0.0f, FColor::White,
        FString::Printf(TEXT("Mode: %s"), *ControlModeStr));

    // Show max velocity and boost status (for Altitude Hold mode)
    if (NMPCControlMode == ENMPCControlMode::AltitudeHold)
    {
        FColor VelocityColor = bSpeedBoostActive ? FColor::Red : FColor::Green;
        FString BoostIndicator = bSpeedBoostActive ? TEXT(" [BOOST]") : TEXT("");
        GEngine->AddOnScreenDebugMessage(52, 0.0f, VelocityColor,
            FString::Printf(TEXT("Max Velocity: %.0f cm/s (%.1f m/s)%s"),
                NMPCAltitudeHoldMaxVelocity, NMPCAltitudeHoldMaxVelocity * 0.01f, *BoostIndicator));

        if (bSpeedBoostActive)
        {
            GEngine->AddOnScreenDebugMessage(53, 0.0f, FColor::Yellow,
                FString::Printf(TEXT("Base Velocity: %.0f cm/s"), BaseMaxVelocity));
        }
    }

    GEngine->AddOnScreenDebugMessage(54, 0.0f, FColor::Green,
        FString::Printf(TEXT("Target Pos: %.1f, %.1f, %.1f"),
            NMPCTargetPosition.X, NMPCTargetPosition.Y, NMPCTargetPosition.Z));

    GEngine->AddOnScreenDebugMessage(55, 0.0f, FColor::Yellow,
        FString::Printf(TEXT("Position Error: %.1f cm"), PositionError.Size()));

    GEngine->AddOnScreenDebugMessage(56, 0.0f, FColor::White,
        FString::Printf(TEXT("Thrust: %.2f N"), CachedNMPCThrust));

    GEngine->AddOnScreenDebugMessage(57, 0.0f, FColor::Magenta,
        FString::Printf(TEXT("Target Angular Vel: R=%.1f P=%.1f Y=%.1f deg/s"),
            CachedNMPCAngularVelocity.X, CachedNMPCAngularVelocity.Y, CachedNMPCAngularVelocity.Z));

    GEngine->AddOnScreenDebugMessage(58, 0.0f, FColor::Orange,
        FString::Printf(TEXT("Solve Time: %.2f ms, Iterations: %d"),
            NMPCController->GetLastSolveTimeMs(), NMPCController->GetLastSolverIterations()));
}

// ========== NMPC PUBLIC API ==========

float AAFPVDronePawn::GetNMPCCollectiveThrust() const
{
    return CachedNMPCThrust;
}

FVector AAFPVDronePawn::GetNMPCTargetAngularVelocity() const
{
    return CachedNMPCAngularVelocity;
}

void AAFPVDronePawn::SetNMPCTargetPosition(const FVector& TargetPosition)
{
    NMPCTargetPosition = TargetPosition;
}

void AAFPVDronePawn::SetNMPCTargetPose(const FVector& TargetPosition, const FRotator& TargetRotation)
{
    NMPCTargetPosition = TargetPosition;
    NMPCTargetRotation = TargetRotation;
}

bool AAFPVDronePawn::IsNMPCReady() const
{
    return NMPCController.IsValid() && NMPCController->IsInitialized();
}

// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/FloatingPawnMovement.h"
#include "Camera/CameraComponent.h"
#include "InputActionValue.h"
#include "NMPC/NMPCDroneController.h"
#include "NMPCControlMode.h"
#include "AFPVDronePawn.generated.h"

class UInputMappingContext;
class UInputAction;

UCLASS()
class DRONESVSPROPS_API AAFPVDronePawn : public APawn
{
    GENERATED_BODY()

public:
    AAFPVDronePawn();

protected:
    virtual void BeginPlay() override;

public:
    virtual void Tick(float DeltaTime) override;
    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

    // ========== COMPONENTS ==========

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    class UCapsuleComponent* CapsuleComponent;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    USkeletalMeshComponent* DroneMesh;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    UCameraComponent* Camera;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    UFloatingPawnMovement* MovementComponent;

    // ========== ENHANCED INPUT ==========

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
    UInputMappingContext* InputMappingContext;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
    UInputAction* YawThrottleAction; // Gamepad Left Stick 2D Axis (X=Yaw, Y=Throttle)

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
    UInputAction* RollPitchAction; // Gamepad Right Stick (Roll/Pitch rates)

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
    UInputAction* CameraTiltAction; // D-Pad Up/Down for camera tilt adjustment

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
    UInputAction* MouseLookAction; // Mouse movement for camera control (altitude hold mode)

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
    UInputAction* KeyboardMoveAction; // WASD for movement (altitude hold mode)

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
    UInputAction* MouseWheelAction; // Mouse wheel for adjusting max velocity (altitude hold mode)

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
    UInputAction* SpeedBoostAction; // Shift key for speed boost (altitude hold mode)

    // ========== ACRO MODE PARAMETERS ==========

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates")
    float MaxRollRate = 600.0f; // degrees per second

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates")
    float MaxPitchRate = 600.0f; // degrees per second

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates")
    float MaxYawRate = 500.0f; // degrees per second

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Throttle")
    float MinThrottle = 0.0f; // normalized [0,1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Throttle")
    float MaxThrottle = 1.0f; // normalized [0,1]

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Throttle", meta = (ClampMin = "0.5", ClampMax = "10.0"))
    float ThrottlePowerMultiplier = 3.0f; // Throttle power relative to gravity (3.0 = can climb at max throttle)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Physics")
    float GravityAcceleration = 980.0f; // cm/s^2 (standard gravity)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Physics")
    float AirDrag = 0.5f; // Air resistance coefficient

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Physics")
    float MaxHorizontalSpeed = 20000.0f; // cm/s

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Physics")
    bool bEnableGroundEffect = false; // Enable ground effect cushion near the ground

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Physics", meta = (EditCondition = "bEnableGroundEffect"))
    float GroundEffectHeight = 50.0f; // Height at which ground effect starts (cm)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Physics", meta = (EditCondition = "bEnableGroundEffect"))
    float GroundEffectStrength = 500.0f; // Upward force when near ground (cm/s^2)

    // ========== RATE CONTROL PARAMETERS (Betaflight-style) ==========

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Roll", meta = (ClampMin = "0.0", ClampMax = "2.55"))
    float RollRCRate = 1.0f; // RC Rate: Overall sensitivity multiplier (0.0-2.55, Default: 1.0 for beginners)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Roll", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float RollSuperRate = 0.0f; // Super Rate: Increases max rate at stick ends (0.0-1.0, Default: 0.0 for beginners)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Roll", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float RollExpo = 0.0f; // Expo: Reduces sensitivity at center stick (0.0-1.0, Default: 0.0 for beginners)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Pitch", meta = (ClampMin = "0.0", ClampMax = "2.55"))
    float PitchRCRate = 1.0f; // RC Rate for pitch

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Pitch", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float PitchSuperRate = 0.0f; // Super Rate for pitch

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Pitch", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float PitchExpo = 0.0f; // Expo for pitch

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Yaw", meta = (ClampMin = "0.0", ClampMax = "2.55"))
    float YawRCRate = 1.0f; // RC Rate for yaw

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Yaw", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float YawSuperRate = 0.0f; // Super Rate for yaw

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Rates|Yaw", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float YawExpo = 0.0f; // Expo for yaw

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Control")
    float RateDeadzone = 0.1f; // Stick deadzone (0.0-0.5, Default: 0.1)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Control")
    float AngularDamping = 2.0f; // How quickly rotation slows when stick centered

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Simulation", meta = (ClampMin = "0.0001", ClampMax = "0.01"))
    float PhysicsSubstepTime = 0.002f; // Physics simulation substep time in seconds (0.002 = 500Hz, smaller = more accurate)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ACRO|Throttle")
    float ThrottleResponseCurve = 1.0f; // 1.0 = linear, >1.0 = exponential

    // ========== FPV CAMERA PARAMETERS ==========

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|FPV")
    float DefaultCameraTiltAngle = 20.0f; // Default up-tilt angle (degrees)

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|FPV")
    float MinCameraTiltAngle = -30.0f; // Maximum down-tilt

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|FPV")
    float MaxCameraTiltAngle = 60.0f; // Maximum up-tilt

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|FPV")
    float CameraTiltAdjustmentSpeed = 5.0f; // Degrees per button press

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|FPV")
    FVector CameraOffset = FVector(80.0f, 0.0f, 30.0f); // Camera position relative to drone

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|FPV")
    float CameraFOV = 120.0f; // Wide FOV for FPV feel

    // ========== COLLISION PARAMETERS ==========

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Collision")
    float CollisionDamping = 0.8f; // How much velocity is retained after collision

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Collision")
    bool bEnableCollision = true;

    // ========== DEBUG ==========

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
    bool bShowDebugInfo = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Debug")
    bool bShowDebugVectors = false;

    // ========== NMPC CONTROLLER ==========

    /** Enable NMPC-based control mode */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC")
    bool bEnableNMPCControl = false;

    /** NMPC Control Mode: Position Control or Altitude Hold */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC", meta = (EditCondition = "bEnableNMPCControl"))
    ENMPCControlMode NMPCControlMode = ENMPCControlMode::PositionControl;

    /** Drone mass in kg (for NMPC physics model) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC", meta = (EditCondition = "bEnableNMPCControl"))
    float DroneMass = 1.0f;

    /** NMPC prediction horizon length */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "5", ClampMax = "50"))
    int NMPCHorizonLength = 20;

    /** NMPC prediction time step in seconds */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "0.01", ClampMax = "0.2"))
    float NMPCPredictionDt = 0.05f;

    /** Target position for NMPC control (set via Blueprint or code) - in cm (UE5 units) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC", meta = (EditCondition = "bEnableNMPCControl"))
    FVector NMPCTargetPosition = FVector(0.0f, 0.0f, 0.0f); 

    /** Target orientation for NMPC control */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC", meta = (EditCondition = "bEnableNMPCControl"))
    FRotator NMPCTargetRotation = FRotator::ZeroRotator;

    /** Speed at which joystick moves the target position (cm/s) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "10.0", ClampMax = "1000.0"))
    float NMPCTargetMoveSpeed = 300.0f;

    /** Speed at which joystick rotates the target yaw (deg/s) - Position Control Mode */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "10.0", ClampMax = "180.0"))
    float NMPCTargetYawSpeed = 45.0f;

    /** Max horizontal velocity for Altitude Hold mode (cm/s) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "100.0", ClampMax = "2000.0"))
    float NMPCAltitudeHoldMaxVelocity = 500.0f;

    /** Mouse wheel velocity adjustment step (cm/s per wheel notch) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "10.0", ClampMax = "200.0"))
    float MouseWheelVelocityStep = 50.0f;

    /** Speed boost acceleration when holding Shift key (cm/s^2) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold|Speed Boost", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "100.0", ClampMax = "5000.0"))
    float SpeedBoostAcceleration = 500.0f;

    /** Maximum velocity when speed boosting (cm/s) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold|Speed Boost", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "500.0", ClampMax = "5000.0"))
    float SpeedBoostMaxVelocity = 2000.0f;

    /** Camera yaw rotation speed when using left stick X in Altitude Hold mode (deg/s) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "10.0", ClampMax = "180.0"))
    float NMPCAltitudeHoldCameraYawSpeed = 90.0f;

    /** Camera pitch rotation speed when using left stick Y in Altitude Hold mode (deg/s) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "10.0", ClampMax = "180.0"))
    float NMPCAltitudeHoldCameraPitchSpeed = 60.0f;

    /** Min camera pitch angle in Altitude Hold mode (degrees) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "-89.0", ClampMax = "0.0"))
    float NMPCAltitudeHoldMinCameraPitch = -89.0f;

    /** Max camera pitch angle in Altitude Hold mode (degrees) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "0.0", ClampMax = "89.0"))
    float NMPCAltitudeHoldMaxCameraPitch = 89.0f;

    /** Camera rotation interpolation speed for smooth movement */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "1.0", ClampMax = "20.0"))
    float CameraRotationInterpSpeed = 10.0f;

    /** Mouse sensitivity for yaw (X-axis) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold|Mouse", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "0.1", ClampMax = "5.0"))
    float MouseYawSensitivity = 1.0f;

    /** Mouse sensitivity for pitch (Y-axis) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Altitude Hold|Mouse", meta = (EditCondition = "bEnableNMPCControl", ClampMin = "0.1", ClampMax = "5.0"))
    float MousePitchSensitivity = 1.0f;

    /** Show NMPC debug info on screen */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NMPC|Debug", meta = (EditCondition = "bEnableNMPCControl"))
    bool bShowNMPCDebugInfo = false;

    // ========== NMPC PUBLIC API ==========

    /** Get the NMPC computed collective thrust (Newtons) */
    UFUNCTION(BlueprintCallable, Category = "NMPC")
    float GetNMPCCollectiveThrust() const;

    /** Get the NMPC computed target angular velocity (deg/s) */
    UFUNCTION(BlueprintCallable, Category = "NMPC")
    FVector GetNMPCTargetAngularVelocity() const;

    /** Set NMPC target position (Blueprint callable) */
    UFUNCTION(BlueprintCallable, Category = "NMPC")
    void SetNMPCTargetPosition(const FVector& TargetPosition);

    /** Set NMPC target pose (position + rotation) */
    UFUNCTION(BlueprintCallable, Category = "NMPC")
    void SetNMPCTargetPose(const FVector& TargetPosition, const FRotator& TargetRotation);

    /** Check if NMPC controller is ready */
    UFUNCTION(BlueprintCallable, Category = "NMPC")
    bool IsNMPCReady() const;

protected:
    // ========== INPUT HANDLERS ==========

    void OnYawThrottle(const FInputActionValue& Value);
    void OnRollPitch(const FInputActionValue& Value);
    void OnCameraTilt(const FInputActionValue& Value);
    void OnMouseLook(const FInputActionValue& Value);
    void OnKeyboardMove(const FInputActionValue& Value);
    void OnMouseWheel(const FInputActionValue& Value);
    void OnSpeedBoost(const FInputActionValue& Value);

    // ========== ACRO FLIGHT LOGIC ==========

    void UpdateAngularVelocity(float DeltaTime);
    void ApplyRotation(float DeltaTime);
    void ApplyThrottle(float Thrust, float DeltaTime);
    void ApplyAcceleration(FVector Acceleration, float DeltaTime);

    void ApplyGravity(float DeltaTime);
    void ApplyGroundEffect(float DeltaTime);
    void ApplyAirDrag(float DeltaTime);
    void UpdateCameraTilt();
    void ClampVelocity();

    // ========== NMPC CONTROL ==========

    void InitializeNMPCController();
    void UpdateNMPCControl(float DeltaTime);
    void ApplyNMPCControl(float DeltaTime);
    void UpdateNMPCTargetFromInput(float DeltaTime);
    void PrintNMPCDebugInfo();

    // ========== COLLISION ==========

    UFUNCTION()
    void OnHit(UPrimitiveComponent* HitComponent, AActor* OtherActor,
        UPrimitiveComponent* OtherComp, FVector NormalImpulse, const FHitResult& Hit);

    // ========== DEBUG ==========

    void DrawDebugVectors();
    void PrintDebugInfo();

    // ========== UTILITY ==========

    float ApplyCurve(float Input, float Curve) const;
    float ApplyDeadzone(float Input, float Deadzone) const;
    float ApplyBetaflightRates(float Input, float RCRate, float SuperRate, float Expo) const;

private:
    // ========== INPUT STATE ==========

    // Raw input from gamepad [-1, 1]
    float ThrottleInput = 0.0f;      // Left Trigger (0 to 1)
    float ThrottleInputRaw = 0.0f;   // Left Stick Y (-1 to 1)
    float RollInput = 0.0f;          // Right Stick X (processed with Betaflight rates)
    float RollInputRaw = 0.0f;       // Right Stick X raw [-1, 1]
    float PitchInput = 0.0f;         // Right Stick Y (processed with Betaflight rates)
    float PitchInputRaw = 0.0f;      // Right Stick Y raw [-1, 1]
    float YawInput = 0.0f;           // Left Stick X
    float YawInputRaw = 0.0f;        // Left Stick X raw [-1, 1]

    // Mouse and keyboard input (for altitude hold mode)
    FVector2D MouseDelta = FVector2D::ZeroVector;  // Mouse movement delta
    FVector2D KeyboardMoveInput = FVector2D::ZeroVector;  // WASD input [-1, 1]

    // Speed boost state
    bool bSpeedBoostActive = false;  // Is speed boost currently active?
    float BaseMaxVelocity = 500.0f;  // Velocity when boost started

    // ========== FLIGHT STATE ==========

    // Angular velocities (degrees per second)
    FVector AngularVelocity = FVector::ZeroVector; // X=Roll, Y=Pitch, Z=Yaw

    // Current throttle value [0, 1]
    float CurrentThrottle = 0.0f;

    // Current camera tilt angle (degrees)
    float CurrentCameraTilt = 0.0f;

    // Velocity tracking for physics
    FVector CurrentVelocity = FVector::ZeroVector;

    // ========== NMPC STATE ==========

    /** NMPC Controller instance */
    TUniquePtr<FNMPCDroneController> NMPCController;

    /** Cached NMPC outputs */
    float CachedNMPCThrust = 0.0f;
    FVector CachedNMPCAngularVelocity = FVector::ZeroVector;

    /** Manual physics state for NMPC mode (bypasses FloatingPawnMovement) */
    FVector NMPCVelocity = FVector::ZeroVector; // cm/s
    FVector NMPCPosition = FVector::ZeroVector; // cm

    /** Target velocity for Altitude Hold mode (cm/s) */
    FVector NMPCTargetVelocity = FVector::ZeroVector;

    /** Target control rotation for altitude hold camera control */
    FRotator TargetControlRotation = FRotator::ZeroRotator;

    /** Current smoothed control rotation */
    FRotator CurrentControlRotation = FRotator::ZeroRotator;
};
