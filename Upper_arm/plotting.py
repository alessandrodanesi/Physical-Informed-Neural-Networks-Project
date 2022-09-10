import matplotlib.pyplot as plt
import numpy as np

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True)

pred_vessel_1= np.load("../../upper arm 50k -15k/Prediction_Vessel1.npy", allow_pickle=True).item()
test_vessel_1 = np.load("./Inputs/test_1.npy", allow_pickle=True).item()

pressure_test_vessel1 = test_vessel_1["Pressure"][:, None]*1e06
pressure_pred_vessel_1 = pred_vessel_1["Pressure"]*1e03
t = test_vessel_1["Time"][:, None]
T = pred_vessel_1["Time"]

#fig2 = plt.figure(2, figsize=(22, 12), dpi=110, facecolor='w', frameon=False)
plt.plot(t, pressure_test_vessel1, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
plt.plot(T, pressure_pred_vessel_1, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
plt.show()





pred_vessel_2= np.load("../../upper arm 50k -15k/Prediction_Vessel2.npy", allow_pickle=True).item()
test_vessel_2 = np.load("./Inputs/test_2.npy", allow_pickle=True).item()

pressure_test_vessel2 = test_vessel_2["Pressure"][:, None]*1e06
pressure_pred_vessel_2 = pred_vessel_2["Pressure"]*1e03
t = test_vessel_2["Time"][:, None]
T = pred_vessel_2["Time"]

#fig2 = plt.figure(2, figsize=(22, 12), dpi=110, facecolor='w', frameon=False)
plt.plot(t, pressure_test_vessel2, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
plt.plot(T, pressure_pred_vessel_2, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
plt.show()



pred_vessel_3= np.load("../../upper arm 50k -15k/Prediction_Vessel3.npy", allow_pickle=True).item()
test_vessel_3 = np.load("./Inputs/test_3.npy", allow_pickle=True).item()

pressure_test_vessel3 = test_vessel_3["Pressure"][:, None]*1e06
pressure_pred_vessel_3 = pred_vessel_3["Pressure"]*1e03
t = test_vessel_3["Time"][:, None]
T = pred_vessel_3["Time"]

#fig2 = plt.figure(2, figsize=(22, 12), dpi=110, facecolor='w', frameon=False)
plt.plot(t, pressure_test_vessel3, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
plt.plot(T, pressure_pred_vessel_3, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
plt.suptitle('Comparative pressure Vessel 3')
plt.xlabel("t in s")
plt.ylabel("Pressure in Pa")
plt.legend(loc='upper right', frameon=False, fontsize = 'medium')
plt.show()




pred_vessel_4= np.load("../../upper arm 50k -15k/Prediction_Vessel4.npy", allow_pickle=True).item()
test_vessel_4 = np.load("./Inputs/test_4.npy", allow_pickle=True).item()

pressure_test_vessel4 = test_vessel_4["Pressure"][:, None]*1e06
pressure_pred_vessel_4 = pred_vessel_4["Pressure"]*1e03
t = test_vessel_4["Time"][:, None]
T = pred_vessel_4["Time"]

#fig2 = plt.figure(2, figsize=(22, 12), dpi=110, facecolor='w', frameon=False)
plt.plot(t, pressure_test_vessel4, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
plt.plot(T, pressure_pred_vessel_4, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
plt.show()



pred_vessel_5= np.load("../../upper arm 50k -15k/Prediction_Vessel5.npy", allow_pickle=True).item()
test_vessel_5 = np.load("./Inputs/test_5.npy", allow_pickle=True).item()

pressure_test_vessel5 = test_vessel_5["Pressure"][:, None]*1e06
pressure_pred_vessel_5 = pred_vessel_5["Pressure"]*1e03
t = test_vessel_5["Time"][:, None]
T = pred_vessel_5["Time"]

#fig2 = plt.figure(2, figsize=(22, 12), dpi=110, facecolor='w', frameon=False)
plt.plot(t, pressure_test_vessel5, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
plt.plot(T, pressure_pred_vessel_5, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
plt.show()




pred_vessel_6= np.load("../../upper arm 50k -15k/Prediction_Vessel6.npy", allow_pickle=True).item()
test_vessel_6 = np.load("./Inputs/test_6.npy", allow_pickle=True).item()

pressure_test_vessel6 = test_vessel_6["Pressure"][:, None]*1e06
pressure_pred_vessel_6 = pred_vessel_6["Pressure"]*1e03
t = test_vessel_6["Time"][:, None]
T = pred_vessel_6["Time"]

#fig2 = plt.figure(2, figsize=(22, 12), dpi=110, facecolor='w', frameon=False)
plt.plot(t, pressure_test_vessel6, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
plt.plot(T, pressure_pred_vessel_6, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
plt.show()




pred_vessel_7= np.load("../../upper arm 50k -15k/Prediction_Vessel7.npy", allow_pickle=True).item()
test_vessel_7 = np.load("./Inputs/test_7.npy", allow_pickle=True).item()

pressure_test_vessel7 = test_vessel_7["Pressure"][:, None]*1e06
pressure_pred_vessel_7 = pred_vessel_7["Pressure"]*1e03
t = test_vessel_7["Time"][:, None]
T = pred_vessel_7["Time"]

#fig2 = plt.figure(2, figsize=(22, 12), dpi=110, facecolor='w', frameon=False)
plt.plot(t, pressure_test_vessel7, 'r--', linewidth=1, markersize=0.5, label='Reference pressure Vessel1')
plt.plot(T, pressure_pred_vessel_7, 'b--', linewidth=1, markersize=0.5, label='Predicted pressure Vessel1')
plt.show()