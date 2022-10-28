from sklearn.utils import shuffle
from src.utils import read_file, compute_metrics
from src.FLH import FLH, FLH_Revised
from src.AAR import AAR
from src.preprocessing import timestamp_parsing, clean_nan
import matplotlib.pyplot as plt

DATASET_PATH = "./dataset/data.csv"

if __name__ == "__main__":
    
    df = read_file(DATASET_PATH, None)
    x_cols = ['Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4']
    y_cols = ['Measurement']
    
    # data preprocessing
    # convert object to timestamp
    timestamp_parsing(df, col = 'Time', is_parsing = False)
    
    # remove nan or null data
    clean_nan(df)
    
    # visualization
    fig, axes = plt.subplots(4, figsize = (8,12), sharex = True)
    for ax, x_col in zip(axes,x_cols):
        ax.plot(df['Time'], df[x_col], label = x_col)
        ax.legend(loc = 'upper right')
        ax.set_ylabel(x_col)
    ax.set_xlabel('time')
    plt.savefig("./result/trend.png")
    
    # step 1. Agregating Algorithm for Regression(AAR)
    x,y = df[x_cols].values, df[y_cols].values
    
    model_aar = AAR(input_dims = 4, gamma = 0.1)
    y_aar = model_aar.fit_predict(x,y)
    plt.figure(2, figsize = (10,6))
    plt.plot(df['Time'], y_aar, 'ro-', label = "prediction")
    plt.plot(df['Time'], y, "b-", label = "actual")
    plt.xlabel("Time")
    plt.ylabel("Measurement")
    plt.legend()
    plt.savefig("./result/AAR.png")
    
    
    
    # step 2. Follow-Leading-History algorithms(FLH) with Agregating Algorithms for Regression(AAR)
    model_FLH = FLH(alpha = 0.1, gamma = 0.1, input_dims = 4)
    y_flh = model_FLH.fit_predict(x, y)
    
    plt.figure(3, figsize = (10,6))
    plt.plot(df['Time'], y_flh, 'ro-', label = "prediction")
    plt.plot(df['Time'], y, "b-", label = "actual")
    plt.xlabel("Time")
    plt.ylabel("Measurement")
    plt.legend()
    plt.savefig("./result/FLH.png")
    
    # step 3. Follow-Leading-History algorithms with revised (FLH) with Agregating Algorithms for Regression(AAR)
    model_FLH_revised = FLH_Revised(alpha = 0.1, gamma = 0.1, input_dims = 4)
    y_flh_revised = model_FLH_revised.fit_predict(x, y)
    
    plt.figure(4, figsize = (10,6))
    plt.plot(df['Time'], y_flh_revised, 'ro-', label = "prediction")
    plt.plot(df['Time'], y, "b-", label = "actual")
    plt.xlabel("Time")
    plt.ylabel("Measurement")
    plt.legend()
    plt.savefig("./result/FLH_revised.png")
    
    
    print("="*24," Result ","="*24)
    compute_metrics(y, y_aar, algorithm = "AAR")
    compute_metrics(y, y_flh, algorithm = "FLH")
    compute_metrics(y, y_flh_revised, algorithm = "FLH-Revised")