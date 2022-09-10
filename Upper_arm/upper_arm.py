import matplotlib.pyplot as plt
import numpy as np

from upper_arm_pinns import OneDBioPINN

if __name__ == "__main__":
    N_f = 2000
    
    ################## BACKUP #####################
    path = ''
    #restore_backup = './Model/' + path  # specify where to load the backup
    restore_backup = 'NO'  # uncomment to NOT load a backup

    backup_file = restore_backup + 'final model'  # not used if restore_backup = 'NO', just leave it like it is....
    
    ####### MODEL UPPER ARM (see picture) ########
    # 1) Axilliary R
    # 2) Brachial R
    # 3) R. Radial
    # 4) R. Ulnar 1
    # 5) R. Intrerosseus
    # 6) Posterior Interosseus R
    # 7) R. Ulnar 2

    # # -------  BIFURCATIONS  ------- #
    # bif_vessels = [[1, 2, 3],
    #                [3, 4, 6]]
    #
    # # -------  CONJUCTIONS  ------- #
    # conj_points = [[0, 1],
    #                [4, 5]]

    # upload 7 upper arm vessels data (pressure, velocity, area) in a specific point x for each vessel
    # In particular, I divided each vessels in 1001 points. So the middle point is the 500th in the array and I
    # chose this one as data in input. For test instead, I chose 800th point in each vessel.


    # Notice that:
    # Time interval is about 0.8s (1 cycle)
    # Pressure in Mpa. To have Pa multiply by 10e6
    # Velocity in mm/s. To have m/s divide by 1000
    # Area in mm^2. To have m^2 multiply by 1E-06
    # Flow in mm^3/s. To have m^3/s multiply by 1E-09

    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True)

    input_vessel_1 = np.load("./Inputs/input_1.npy", allow_pickle=True).item()
    input_vessel_2 = np.load("./Inputs/input_2.npy", allow_pickle=True).item()
    input_vessel_3 = np.load("./Inputs/input_3.npy", allow_pickle=True).item()
    input_vessel_4 = np.load("./Inputs/input_4.npy", allow_pickle=True).item()
    input_vessel_5 = np.load("./Inputs/input_5.npy", allow_pickle=True).item()
    input_vessel_6 = np.load("./Inputs/input_6.npy", allow_pickle=True).item()
    input_vessel_7 = np.load("./Inputs/input_7.npy", allow_pickle=True).item()

    t = input_vessel_1["Time"][:, None]

    # as before,but these data will be used for testing
    test_vessel_1 = np.load("./Inputs/test_1.npy", allow_pickle=True).item()
    test_vessel_2 = np.load("./Inputs/test_2.npy", allow_pickle=True).item()
    test_vessel_3 = np.load("./Inputs/test_3.npy", allow_pickle=True).item()
    test_vessel_4 = np.load("./Inputs/test_4.npy", allow_pickle=True).item()
    test_vessel_5 = np.load("./Inputs/test_5.npy", allow_pickle=True).item()
    test_vessel_6 = np.load("./Inputs/test_6.npy", allow_pickle=True).item()
    test_vessel_7 = np.load("./Inputs/test_7.npy", allow_pickle=True).item()

    # take velocities from INPUT data
    velocity_measurements_vessel1 = input_vessel_1["Velocity"][:, None]*1e-03
    velocity_measurements_vessel2 = input_vessel_2["Velocity"][:, None]*1e-03
    velocity_measurements_vessel3 = input_vessel_3["Velocity"][:, None]*1e-03
    velocity_measurements_vessel4 = input_vessel_4["Velocity"][:, None]*1e-03
    velocity_measurements_vessel5 = input_vessel_5["Velocity"][:, None]*1e-03
    velocity_measurements_vessel6 = input_vessel_6["Velocity"][:, None]*1e-03
    velocity_measurements_vessel7 = input_vessel_7["Velocity"][:, None]*1e-03

    # take areas from INPUT data
    area_measurements_vessel1 = input_vessel_1["Area"][:, None]*1e-06
    area_measurements_vessel2 = input_vessel_2["Area"][:, None]*1e-06
    area_measurements_vessel3 = input_vessel_3["Area"][:, None]*1e-06
    area_measurements_vessel4 = input_vessel_4["Area"][:, None]*1e-06
    area_measurements_vessel5 = input_vessel_5["Area"][:, None]*1e-06
    area_measurements_vessel6 = input_vessel_6["Area"][:, None]*1e-06
    area_measurements_vessel7 = input_vessel_7["Area"][:, None]*1e-06


    # take velocities from TEST data
    velocity_test_vessel1 = test_vessel_1["Velocity"][:, None]*1e-03
    velocity_test_vessel2 = test_vessel_2["Velocity"][:, None]*1e-03
    velocity_test_vessel3 = test_vessel_3["Velocity"][:, None]*1e-03
    velocity_test_vessel4 = test_vessel_4["Velocity"][:, None]*1e-03
    velocity_test_vessel5 = test_vessel_5["Velocity"][:, None]*1e-03
    velocity_test_vessel6 = test_vessel_6["Velocity"][:, None]*1e-03
    velocity_test_vessel7 = test_vessel_7["Velocity"][:, None]*1e-03

    # take pressures from TEST data in Pa
    pressure_test_vessel1 = test_vessel_1["Pressure"][:, None]*1e06
    pressure_test_vessel2 = test_vessel_2["Pressure"][:, None]*1e06
    pressure_test_vessel3 = test_vessel_3["Pressure"][:, None]*1e06
    pressure_test_vessel4 = test_vessel_4["Pressure"][:, None]*1e06
    pressure_test_vessel5 = test_vessel_5["Pressure"][:, None]*1e06
    pressure_test_vessel6 = test_vessel_6["Pressure"][:, None]*1e06
    pressure_test_vessel7 = test_vessel_7["Pressure"][:, None]*1e06

    
    # restore np.load for future normal usage
    np.load = np_load_old
    # INITIALIZE VARIABLES FOR TRAINING ------------------------------------------------------------------------

    N_u = t.shape[0]

    layers = [2, 100, 100, 100, 100, 100, 100, 100, 3]

    lower_bound_t = t.min(0)
    upper_bound_t = t.max(0)

    # I consider lengths in m now (not like in the pylsewave package)

    lower_bound_vessel_1 = 0.0
    upper_bound_vessel_1 = 0.12

    lower_bound_vessel_2 = upper_bound_vessel_1
    upper_bound_vessel_2 = upper_bound_vessel_1 + 0.22311

    lower_bound_vessel_3 = upper_bound_vessel_2
    upper_bound_vessel_3 = upper_bound_vessel_2 + 0.30089

    lower_bound_vessel_4 = upper_bound_vessel_2
    upper_bound_vessel_4 = upper_bound_vessel_2 + 0.02976

    lower_bound_vessel_5 = upper_bound_vessel_4
    upper_bound_vessel_5 = upper_bound_vessel_4 + 0.01627

    lower_bound_vessel_6 = upper_bound_vessel_5
    upper_bound_vessel_6 = upper_bound_vessel_5 + 0.23056

    lower_bound_vessel_7 = upper_bound_vessel_4
    upper_bound_vessel_7 = upper_bound_vessel_4 + 0.23926

    lengths = [upper_bound_vessel_1 - lower_bound_vessel_1,
               upper_bound_vessel_2 - lower_bound_vessel_2,
               upper_bound_vessel_3 - lower_bound_vessel_3,
               upper_bound_vessel_4 - lower_bound_vessel_4,
               upper_bound_vessel_5 - lower_bound_vessel_5,
               upper_bound_vessel_6 - lower_bound_vessel_6,
               upper_bound_vessel_7 - lower_bound_vessel_7]

    # bif. points
    bif_points = [upper_bound_vessel_2, upper_bound_vessel_4]  # 2 bif

    # Initial coordinates
    X_initial_vessel1 = np.linspace(lower_bound_vessel_1, upper_bound_vessel_1, N_u)[:, None]
    X_initial_vessel2 = np.linspace(lower_bound_vessel_2, upper_bound_vessel_2, N_u)[:, None]
    X_initial_vessel3 = np.linspace(lower_bound_vessel_3, upper_bound_vessel_3, N_u)[:, None]
    X_initial_vessel4 = np.linspace(lower_bound_vessel_4, upper_bound_vessel_4, N_u)[:, None]
    X_initial_vessel5 = np.linspace(lower_bound_vessel_5, upper_bound_vessel_5, N_u)[:, None]
    X_initial_vessel6 = np.linspace(lower_bound_vessel_6, upper_bound_vessel_6, N_u)[:, None]
    X_initial_vessel7 = np.linspace(lower_bound_vessel_7, upper_bound_vessel_7, N_u)[:, None]

    # Measurement points -> middle of each vessel (500th point since the array has length 1001)
    measurement_points = [X_initial_vessel1[500].item(),
                          X_initial_vessel2[500].item(),
                          X_initial_vessel3[500].item(),
                          X_initial_vessel4[500].item(),
                          X_initial_vessel5[500].item(),
                          X_initial_vessel6[500].item(),
                          X_initial_vessel7[500].item()]

    # Test points -> 800th of each vessel (it's random, but you have to choose it according to data)
    test_points = [X_initial_vessel1[800].item(),
                   X_initial_vessel2[800].item(),
                   X_initial_vessel3[800].item(),
                   X_initial_vessel4[800].item(),
                   X_initial_vessel5[800].item(),
                   X_initial_vessel6[800].item(),
                   X_initial_vessel7[800].item()]

    T_initial = lower_bound_t * np.ones((N_u))[:, None]

    # Measurements points (called boundary I don't know why)
    X_boundary_vessel1 = measurement_points[0] * np.ones((N_u))[:, None]
    X_boundary_vessel2 = measurement_points[1] * np.ones((N_u))[:, None]
    X_boundary_vessel3 = measurement_points[2] * np.ones((N_u))[:, None]
    X_boundary_vessel4 = measurement_points[3] * np.ones((N_u))[:, None]
    X_boundary_vessel5 = measurement_points[4] * np.ones((N_u))[:, None]
    X_boundary_vessel6 = measurement_points[5] * np.ones((N_u))[:, None]
    X_boundary_vessel7 = measurement_points[6] * np.ones((N_u))[:, None]

    T_boundary = t

    # stack Initial + Measurements x_points
    X_measurement_vessel1 = np.vstack((X_initial_vessel1, X_boundary_vessel1))
    X_measurement_vessel2 = np.vstack((X_initial_vessel2, X_boundary_vessel2))
    X_measurement_vessel3 = np.vstack((X_initial_vessel3, X_boundary_vessel3))
    X_measurement_vessel4 = np.vstack((X_initial_vessel4, X_boundary_vessel4))
    X_measurement_vessel5 = np.vstack((X_initial_vessel5, X_boundary_vessel5))
    X_measurement_vessel6 = np.vstack((X_initial_vessel6, X_boundary_vessel6))
    X_measurement_vessel7 = np.vstack((X_initial_vessel7, X_boundary_vessel7))

    T_measurement = np.vstack((T_initial, T_boundary))

    # X Residual points
    X_residual_vessel1 = lower_bound_vessel_1 + (upper_bound_vessel_1 - lower_bound_vessel_1) * np.random.random((N_f))[
                                                                                                :, None]
    X_residual_vessel2 = lower_bound_vessel_2 + (upper_bound_vessel_2 - lower_bound_vessel_2) * np.random.random((N_f))[
                                                                                                :, None]
    X_residual_vessel3 = lower_bound_vessel_3 + (upper_bound_vessel_3 - lower_bound_vessel_3) * np.random.random((N_f))[
                                                                                                :, None]
    X_residual_vessel4 = lower_bound_vessel_4 + (upper_bound_vessel_4 - lower_bound_vessel_4) * np.random.random((N_f))[
                                                                                                :, None]
    X_residual_vessel5 = lower_bound_vessel_5 + (upper_bound_vessel_5 - lower_bound_vessel_5) * np.random.random((N_f))[
                                                                                                :, None]
    X_residual_vessel6 = lower_bound_vessel_6 + (upper_bound_vessel_6 - lower_bound_vessel_6) * np.random.random((N_f))[
                                                                                                :, None]
    X_residual_vessel7 = lower_bound_vessel_7 + (upper_bound_vessel_7 - lower_bound_vessel_7) * np.random.random((N_f))[
                                                                                                :, None]

    T_residual = lower_bound_t + (upper_bound_t - lower_bound_t) * np.random.random((N_f))[:, None]

    A_initial_vessel1 = area_measurements_vessel1[0, 0] * np.ones((N_u, 1))
    A_initial_vessel2 = area_measurements_vessel2[0, 0] * np.ones((N_u, 1))
    A_initial_vessel3 = area_measurements_vessel3[0, 0] * np.ones((N_u, 1))
    A_initial_vessel4 = area_measurements_vessel4[0, 0] * np.ones((N_u, 1))
    A_initial_vessel5 = area_measurements_vessel5[0, 0] * np.ones((N_u, 1))
    A_initial_vessel6 = area_measurements_vessel6[0, 0] * np.ones((N_u, 1))
    A_initial_vessel7 = area_measurements_vessel7[0, 0] * np.ones((N_u, 1))

    u_initial_vessel1 = velocity_measurements_vessel1[0, 0] * np.ones((N_u, 1))
    u_initial_vessel2 = velocity_measurements_vessel2[0, 0] * np.ones((N_u, 1))
    u_initial_vessel3 = velocity_measurements_vessel3[0, 0] * np.ones((N_u, 1))
    u_initial_vessel4 = velocity_measurements_vessel4[0, 0] * np.ones((N_u, 1))
    u_initial_vessel5 = velocity_measurements_vessel5[0, 0] * np.ones((N_u, 1))
    u_initial_vessel6 = velocity_measurements_vessel6[0, 0] * np.ones((N_u, 1))
    u_initial_vessel7 = velocity_measurements_vessel7[0, 0] * np.ones((N_u, 1))

    A_training_vessel1 = np.vstack((A_initial_vessel1, area_measurements_vessel1))
    u_training_vessel1 = np.vstack((u_initial_vessel1, velocity_measurements_vessel1))

    A_training_vessel2 = np.vstack((A_initial_vessel2, area_measurements_vessel2))
    u_training_vessel2 = np.vstack((u_initial_vessel2, velocity_measurements_vessel2))

    A_training_vessel3 = np.vstack((A_initial_vessel3, area_measurements_vessel3))
    u_training_vessel3 = np.vstack((u_initial_vessel3, velocity_measurements_vessel3))

    A_training_vessel4 = np.vstack((A_initial_vessel4, area_measurements_vessel4))
    u_training_vessel4 = np.vstack((u_initial_vessel4, velocity_measurements_vessel4))

    A_training_vessel5 = np.vstack((A_initial_vessel5, area_measurements_vessel5))
    u_training_vessel5 = np.vstack((u_initial_vessel5, velocity_measurements_vessel5))

    A_training_vessel6 = np.vstack((A_initial_vessel6, area_measurements_vessel6))
    u_training_vessel6 = np.vstack((u_initial_vessel6, velocity_measurements_vessel6))

    A_training_vessel7 = np.vstack((A_initial_vessel7, area_measurements_vessel7))
    u_training_vessel7 = np.vstack((u_initial_vessel7, velocity_measurements_vessel7))
    
    ################################## INITIALIZE THE NETWORK #############################

    if restore_backup != 'NO':
        # if you're loading a backup... load the backup

        # save np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

        param = np.load(restore_backup + 'Collocation_Points.npy').item()

        # restore np.load for future normal usage
        np.load = np_load_old

        X_residual_vessel1 = param['X_residual_vessel1']
        X_residual_vessel2 = param['X_residual_vessel2']
        X_residual_vessel3 = param['X_residual_vessel3']
        X_residual_vessel4 = param['X_residual_vessel4']
        X_residual_vessel5 = param['X_residual_vessel5']
        X_residual_vessel6 = param['X_residual_vessel6']
        X_residual_vessel7 = param['X_residual_vessel7']
        T_residual = param['T_residual']
        del param

    else:
        # if you're not loading a backup... save for backup
        param = dict({'X_residual_vessel1': X_residual_vessel1, 'X_residual_vessel2': X_residual_vessel2,
                      'X_residual_vessel3': X_residual_vessel3, 'X_residual_vessel4': X_residual_vessel4,
                      'X_residual_vessel5': X_residual_vessel5, 'X_residual_vessel6': X_residual_vessel6,
                      'X_residual_vessel7': X_residual_vessel7, 'T_residual': T_residual})
        np.save('./Model/' + path + 'Collocation_Points.npy', param)
        del param

    model = OneDBioPINN(X_measurement_vessel1,
                        X_measurement_vessel2,
                        X_measurement_vessel3,
                        X_measurement_vessel4,
                        X_measurement_vessel5,
                        X_measurement_vessel6,
                        X_measurement_vessel7,
                        A_training_vessel1, u_training_vessel1,
                        A_training_vessel2, u_training_vessel2,
                        A_training_vessel3, u_training_vessel3,
                        A_training_vessel4, u_training_vessel4,
                        A_training_vessel5, u_training_vessel5,
                        A_training_vessel6, u_training_vessel6,
                        A_training_vessel7, u_training_vessel7,
                        X_residual_vessel1,
                        X_residual_vessel2,
                        X_residual_vessel3,
                        X_residual_vessel4,
                        X_residual_vessel5,
                        X_residual_vessel6,
                        X_residual_vessel7,
                        T_residual, T_measurement, layers, bif_points, T_initial, lengths)

    ##################### TRAINING OR LOADING ########################

    if restore_backup != 'NO':
        model.load_NN(backup_file,path)
        Total_loss, loss_area, loss_velo, loss_res, loss_cont = model.load_losses(path)

    # Train the neural network
    model.train(50000, 1e-3)
    model.train(15000, 1e-4)

    # Save the trained neural network
    model.save_NN(path) #save_param = 0 if you don't want to save parameters
        


    # EVALUATE, SAVE, AND PLOT PREDICTIONS -----------------------------------------------------------------------------

    # Set folder for saving plots
    path_figures = './Figures/' + path

    # Order T_Residuals
    T_residual.sort(axis=0)

    X_test_vessel1 = test_points[0] * np.ones((T_residual.shape[0], 1))
    X_test_vessel2 = test_points[1] * np.ones((T_residual.shape[0], 1))
    X_test_vessel3 = test_points[2] * np.ones((T_residual.shape[0], 1))
    X_test_vessel4 = test_points[3] * np.ones((T_residual.shape[0], 1))
    X_test_vessel5 = test_points[4] * np.ones((T_residual.shape[0], 1))
    X_test_vessel6 = test_points[5] * np.ones((T_residual.shape[0], 1))
    X_test_vessel7 = test_points[6] * np.ones((T_residual.shape[0], 1))

    A_predicted_vessel1, U_predicted_vessel1, p_predicted_vessel1 = model.predict_vessel1(X_test_vessel1, T_residual, path + 'Prediction_Vessel1')
    A_predicted_vessel2, U_predicted_vessel2, p_predicted_vessel2 = model.predict_vessel2(X_test_vessel2, T_residual, path + 'Prediction_Vessel2')
    A_predicted_vessel3, U_predicted_vessel3, p_predicted_vessel3 = model.predict_vessel3(X_test_vessel3, T_residual, path + 'Prediction_Vessel3')
    A_predicted_vessel4, U_predicted_vessel4, p_predicted_vessel4 = model.predict_vessel4(X_test_vessel4, T_residual, path + 'Prediction_Vessel4')
    A_predicted_vessel5, U_predicted_vessel5, p_predicted_vessel5 = model.predict_vessel5(X_test_vessel5, T_residual, path + 'Prediction_Vessel5')
    A_predicted_vessel6, U_predicted_vessel6, p_predicted_vessel6 = model.predict_vessel6(X_test_vessel6, T_residual, path + 'Prediction_Vessel6')
    A_predicted_vessel7, U_predicted_vessel7, p_predicted_vessel7 = model.predict_vessel7(X_test_vessel7, T_residual, path + 'Prediction_Vessel7')





 # Plot VELOCITY comparision between prediction and reference measurements in 7 vessels
    fig1 = plt.figure(1, figsize=(22, 12), dpi=111, facecolor='w', frameon=False)
    fig2 = plt.figure(2, figsize=(22, 12), dpi=110, facecolor='w', frameon=False)

    ax11 = fig1.add_subplot(241)
    ax12 = fig1.add_subplot(242)
    ax13 = fig1.add_subplot(243)
    ax14 = fig1.add_subplot(244)
    ax15 = fig1.add_subplot(245)
    ax16 = fig1.add_subplot(246)
    ax17 = fig1.add_subplot(247)

    ax21 = fig2.add_subplot(241)
    ax22 = fig2.add_subplot(242)
    ax23 = fig2.add_subplot(243)
    ax24 = fig2.add_subplot(244)
    ax25 = fig2.add_subplot(245)
    ax26 = fig2.add_subplot(246)
    ax27 = fig2.add_subplot(247)

    ax11.plot(t, velocity_test_vessel1, 'r--', linewidth=1, markersize=0.5, label='Reference velocity Vessel1')
    ax11.plot(T_residual, U_predicted_vessel1, 'b--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel1')

    ax12.plot(t, velocity_test_vessel2, 'r--', linewidth=1, markersize=0.5, label='Reference velocity Vessel2')
    ax12.plot(T_residual, U_predicted_vessel2, 'b--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel2')

    ax13.plot(t, velocity_test_vessel3, 'r--', linewidth=1, markersize=0.5, label='Reference velocity Vessel3')
    ax13.plot(T_residual, U_predicted_vessel3, 'b--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel3')

    ax14.plot(t, velocity_test_vessel4, 'r--', linewidth=1, markersize=0.5, label='Reference velocity Vessel4')
    ax14.plot(T_residual, U_predicted_vessel4, 'b--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel4')

    ax15.plot(t, velocity_test_vessel5, 'r--', linewidth=1, markersize=0.5, label='Reference velocity Vessel5')
    ax15.plot(T_residual, U_predicted_vessel5, 'b--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel5')

    ax16.plot(t, velocity_test_vessel6, 'r--', linewidth=1, markersize=0.5, label='Reference velocity Vessel6')
    ax16.plot(T_residual, U_predicted_vessel6, 'b--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel6')

    ax17.plot(t, velocity_test_vessel7, 'r--', linewidth=1, markersize=0.5, label='Reference velocity Vessel7')
    ax17.plot(T_residual, U_predicted_vessel7, 'b--', linewidth=1, markersize=0.5, label='Predicted velocity Vessel7')


    fig1.suptitle('Comparative velocity')
    ax15.set_xlabel("t in s")
    ax11.set_ylabel("Velocity in m/s")
    ax15.set_ylabel("Velocity in m/s")

    fig1.savefig(path_figures + "Comparative_Velocity.png")  # Save figure

    # Plot PRESSURE comparision between prediction and reference measurements in 7 vessels

    ax21.plot(t, pressure_test_vessel1, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
    ax21.plot(T_residual, p_predicted_vessel1, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')

    ax22.plot(t, pressure_test_vessel2, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel2')
    ax22.plot(T_residual, p_predicted_vessel2, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel2')

    ax23.plot(t, pressure_test_vessel3, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel3')
    ax23.plot(T_residual, p_predicted_vessel3, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel3')

    ax24.plot(t, pressure_test_vessel4, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel4')
    ax24.plot(T_residual, p_predicted_vessel4, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel4')

    ax25.plot(t, pressure_test_vessel5, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel5')
    ax25.plot(T_residual, p_predicted_vessel5, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel5')

    ax26.plot(t, pressure_test_vessel6, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel6')
    ax26.plot(T_residual, p_predicted_vessel6, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel6')

    ax27.plot(t, pressure_test_vessel7, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel7')
    ax27.plot(T_residual, p_predicted_vessel7, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel7')


    fig2.suptitle('Comparative pressure')
    ax25.set_xlabel("t in s")
    ax21.set_ylabel("Pressure in Pa")
    ax25.set_ylabel("Pressure in Pa")


    fig2.savefig(path_figures + "Comparative_Pressure.png")  # Save figure

    # ################### LOSS #############
    # print(Total_loss)

    # Evaluate the solution at interface points
    X_test_bif_1 = bif_points[0] * np.ones((T_residual.shape[0], 1))
    X_test_bif_2 = bif_points[1] * np.ones((T_residual.shape[0], 1))

    # bif 1
    A_predicted_interface1, U_predicted_interface1, p_predicted_interface1 = \
        model.predict_vessel1(X_test_bif_1, T_residual, path + 'Prediction_Interface1')
    A_predicted_interface2, U_predicted_interface2, p_predicted_interface2 = \
        model.predict_vessel2(X_test_bif_1, T_residual, path + 'Prediction_Interface2')
    A_predicted_interface3_lb, U_predicted_interface3_lb, p_predicted_interface3_lb = \
        model.predict_vessel3(X_test_bif_1, T_residual, path + 'Prediction_Interface3_lb')

    #bif 2
    A_predicted_interface3_ub, U_predicted_interface3_ub, p_predicted_interface3_ub = \
        model.predict_vessel1(X_test_bif_2, T_residual, path + 'Prediction_Interface3_ub')
    A_predicted_interface4, U_predicted_interface4, p_predicted_interface4 = \
        model.predict_vessel2(X_test_bif_2, T_residual, path + 'Prediction_Interface4')
    A_predicted_interface6, U_predicted_interface6, p_predicted_interface6 = \
        model.predict_vessel3(X_test_bif_2, T_residual, path + 'Prediction_Interface6')

####### bif 1 #######
    # Compute flow in vessels
    Q1 = A_predicted_interface1 * U_predicted_interface1
    Q2 = A_predicted_interface2 * U_predicted_interface2
    Q3_lb = A_predicted_interface3_lb * U_predicted_interface3_lb

    Q_in = Q1
    Q_out = Q2 + Q3_lb

    # Compute the momentum in vessels
    p_1 = p_predicted_interface1 + (0.5 * U_predicted_interface1 ** 2)
    p_2 = p_predicted_interface2 + (0.5 * U_predicted_interface2 ** 2)
    p_3_lb = p_predicted_interface3_lb + (0.5 * U_predicted_interface3_lb ** 2)

    # Plot comparision of flow and momentum to check conservation
    fig3 = plt.figure()

    ax31 = fig3.add_subplot(121)
    ax32 = fig3.add_subplot(122)

    ax31.plot(T_residual, Q_in, 'r-', label='Flow of vessel 1')
    ax31.plot(T_residual, Q_out, 'b--', label='Flow of vessel 2 + 3')

    ax32.plot(T_residual, p_1, 'r-', label='Momentum in vessel 1')
    ax32.plot(T_residual, p_2, 'b--', label='Momentum in vessel 2')
    ax32.plot(T_residual, p_3_lb, 'go', label='Momentum in vessel 3')

    ax31.set_xlabel("t in s")
    ax31.set_ylabel("q(t)[m^3/s]")
    ax32.set_xlabel("t in s")
    ax32.set_ylabel("p(t)[Pa]")

    # Save the plot
    fig3.savefig(path_figures + "Conservation_mass_and_momentum_bif1.png")

####### bif 2 #######
    # Compute flow in vessels
    Q3_ub = A_predicted_interface3_ub * U_predicted_interface3_ub
    Q4 = A_predicted_interface4 * U_predicted_interface4
    Q6 = A_predicted_interface6* U_predicted_interface6

    Q_in = Q3_ub
    Q_out = Q4 + Q6

    # Compute the momentum in vessels
    p_3_ub = p_predicted_interface3_ub + (0.5 * U_predicted_interface3_ub ** 2)
    p_4 = p_predicted_interface4 + (0.5 * U_predicted_interface4 ** 2)
    p_6 = p_predicted_interface6 + (0.5 * U_predicted_interface6 ** 2)

    # Plot comparision of flow and momentum to check conservation
    fig4 = plt.figure()

    ax41 = fig4.add_subplot(121)
    ax42 = fig4.add_subplot(122)

    ax41.plot(T_residual, Q_in, 'r-', label='Flow of vessel 3')
    ax41.plot(T_residual, Q_out, 'b--', label='Flow of vessel 4 + 6')

    ax42.plot(T_residual, p_3_ub, 'r-', label='Momentum in vessel 3')
    ax42.plot(T_residual, p_4, 'b--', label='Momentum in vessel 4')
    ax42.plot(T_residual, p_6, 'go', label='Momentum in vessel 6')

    ax41.set_xlabel("t in s")
    ax41.set_ylabel("q(t)[m^3/s]")
    ax42.set_xlabel("t in s")
    ax42.set_ylabel("p(t)[Pa]")

    # Save the plot
    fig4.savefig(path_figures + "Conservation_mass_and_momentum_bif2.png")



    # Save and print the solution at test points
    X_test_vessel1 = test_points[0] * np.ones((t.shape[0], 1))
    X_test_vessel2 = test_points[1] * np.ones((t.shape[0], 1))
    X_test_vessel3 = test_points[2] * np.ones((t.shape[0], 1))
    X_test_vessel4 = test_points[3] * np.ones((t.shape[0], 1))
    X_test_vessel5 = test_points[4] * np.ones((t.shape[0], 1))
    X_test_vessel6 = test_points[5] * np.ones((t.shape[0], 1))
    X_test_vessel7 = test_points[6] * np.ones((t.shape[0], 1))

    A_predicted_test1, U_predicted_test1, p_predicted_test1 = model.predict_vessel1(X_test_vessel1, t,
                                                                                    path + 'Prediction_Test1')
    A_predicted_test2, U_predicted_test2, p_predicted_test2 = model.predict_vessel2(X_test_vessel2, t,
                                                                                    path + 'Prediction_Test2')
    A_predicted_test3, U_predicted_test3, p_predicted_test3 = model.predict_vessel3(X_test_vessel3, t,
                                                                                    path + 'Prediction_Test3')
    A_predicted_test4, U_predicted_test4, p_predicted_test4 = model.predict_vessel4(X_test_vessel4, t,
                                                                                    path + 'Prediction_Test4')
    A_predicted_test5, U_predicted_test5, p_predicted_test5 = model.predict_vessel5(X_test_vessel5, t,
                                                                                    path + 'Prediction_Test5')
    A_predicted_test6, U_predicted_test6, p_predicted_test6 = model.predict_vessel6(X_test_vessel6, t,
                                                                                    path + 'Prediction_Test6')
    A_predicted_test7, U_predicted_test7, p_predicted_test7 = model.predict_vessel7(X_test_vessel7, t,
                                                                                    path + 'Prediction_Test7')

    # Compute L2 errors for table
    error_p1 = np.linalg.norm(pressure_test_vessel1 - p_predicted_test1, 2) / np.linalg.norm(pressure_test_vessel1, 2)
    error_p2 = np.linalg.norm(pressure_test_vessel2 - p_predicted_test2, 2) / np.linalg.norm(pressure_test_vessel2, 2)
    error_p3 = np.linalg.norm(pressure_test_vessel3 - p_predicted_test3, 2) / np.linalg.norm(pressure_test_vessel3, 2)
    error_p4 = np.linalg.norm(pressure_test_vessel4 - p_predicted_test4, 2) / np.linalg.norm(pressure_test_vessel4, 2)
    error_p5 = np.linalg.norm(pressure_test_vessel5 - p_predicted_test5, 2) / np.linalg.norm(pressure_test_vessel5, 2)
    error_p6 = np.linalg.norm(pressure_test_vessel6 - p_predicted_test6, 2) / np.linalg.norm(pressure_test_vessel6, 2)
    error_p7 = np.linalg.norm(pressure_test_vessel7 - p_predicted_test7, 2) / np.linalg.norm(pressure_test_vessel7, 2)

    print('L2 error in vessel 1:', error_p1)
    print('L2 error in vessel 2:', error_p2)
    print('L2 error in vessel 3:', error_p3)
    print('L2 error in vessel 4:', error_p4)
    print('L2 error in vessel 5:', error_p5)
    print('L2 error in vessel 6:', error_p6)
    print('L2 error in vessel 7:', error_p7)
