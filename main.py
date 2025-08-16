# MSIN0209 Financial Research Project
# University College London
# Forecasting Hedge Fund Strategy Returns Using ARIMAX Models and the S&P500: A Time Series Analysis Pre-, En-, and Post- the COVID-19 Pandemic
# @author Candidate: MBZX8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import chi2
import ruptures as rpt
from scipy.stats import f



# I begin with reading in my data
file_path = ("/Users/saphia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/dissertation/data/full index data "
             "set.xlsx")
index_data = pd.read_excel(file_path)
index_data["Numeric Time"] = range(len(index_data))

index_data["Date"] = pd.to_datetime(index_data["Date"], format="%m/%d/%Y")
index_data.set_index("Date", inplace=True)


# Plot of SPX Return for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["SPX Return"], label="SPX Return", color="blue")
plt.xlabel("Time")
plt.ylabel("SPX Return")
plt.title("Time Series Plot of SPX Return")
plt.legend()
plt.show()


# Plot of Convertible Arbitrage Strategy for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Convertible Arbitrage"], label="Convertible Arbitrage Hedge Fund "
                                                                      "Index Return", color="red")
plt.xlabel("Time")
plt.ylabel("Convertible Arbitrage Hedge Fund Index Return")
plt.title("Time Series Plot of Convertible Arbitrage Hedge Fund Index Return")
plt.legend()
plt.show()


# Plot of Equity L/S Strategy for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Equity L/S"], label="Equity L/S Hedge Fund Index Return", color="green")
plt.xlabel("Time")
plt.ylabel("Equity L/S Hedge Fund Index Return")
plt.title("Time Series Plot of Equity L/S Hedge Fund Index Return")
plt.legend()
plt.show()


# Plot of Equity Market Neutral Strategy for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Equity Market Neutral"], label="Equity Market Neutral Hedge Fund Index Return", color="purple")
plt.xlabel("Time")
plt.ylabel("Equity Market Neutral Hedge Fund Index Return")
plt.title("Time Series Plot of Equity Market Neutral Hedge Fund Index Return")
plt.legend()
plt.show()


# Plot of Event Driven Strategy for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Event Driven"], label="Event Driven Hedge Fund Index Return", color="orange")
plt.xlabel("Time")
plt.ylabel("Event Driven Hedge Fund Index Return")
plt.title("Time Series Plot of Event Driven Hedge Fund Index Return")
plt.legend()
plt.show()


# Plot of Fixed Income Arbitrage Strategy for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Fixed Income Arbitrage"], label="Fixed Income Arbitrage Hedge Fund Index Return", color="pink")
plt.xlabel("Time")
plt.ylabel("Fixed Income Arbitrage Hedge Fund Index Return")
plt.title("Time Series Plot of Fixed Income Arbitrage Hedge Fund Index Return")
plt.legend()
plt.show()


# Plot of Global Macro Strategy for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Global Macro"], label="Global Macro Hedge Fund Index Return", color="blue")
plt.xlabel("Time")
plt.ylabel("Global Macro Hedge Fund Index Return")
plt.title("Time Series Plot of Global Macro Hedge Fund Index Return")
plt.legend()
plt.show()


# Plot of Merger Arbitrage Strategy for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Merger Arbitrage"], label="Merger Arbitrage Hedge Fund Index Return", color="red")
plt.xlabel("Time")
plt.ylabel("Merger Arbitrage Hedge Fund Index Return")
plt.title("Time Series Plot of Merger Arbitrage Hedge Fund Index Return")
plt.legend()
plt.show()


# Plot of Multi Strategy for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Multi Strategy"], label="Multi Strategy Hedge Fund Index Return", color="green")
plt.xlabel("Time")
plt.ylabel("Multi Strategy Hedge Fund Index Return")
plt.title("Time Series Plot of Multi Strategy Hedge Fund Index Return")
plt.legend()
plt.show()


# Plot of Hedge Fund Industry Return for preliminary analysis
plt.figure(figsize=(12,5))
plt.plot(index_data.index, index_data["Barclay Hedge Fund Index"], label="Barclay Hedge Fund Industry Index Return", color="purple")
plt.xlabel("Time")
plt.ylabel("Barclay Hedge Fund Industry Index Return")
plt.title("Time Series Plot of Barclay Hedge Fund Industry Index Return")
plt.legend()
plt.show()


# To check for stationarity, I perform the ADF test.
def adf_test(series, str_name):
    arr_result = adfuller(series, autolag="AIC")
    print(f"ADF Test for {str_name}:")
    print(f"Test Statistic: {arr_result[0]:.4f}")
    print(f"p-value: {arr_result[1]:.4f}")
    print("Critical Values:", arr_result[4])
    print("Stationary" if arr_result[1] < 0.05 else "Non-Stationary", "\n")


adf_test(index_data["SPX Return"], "SPX Return")
adf_test(index_data["Convertible Arbitrage"], "Convertible Arbitrage Hedge Fund Index")
adf_test(index_data["Equity L/S"], "Equity L/S Hedge Fund Index")
adf_test(index_data["Equity Market Neutral"], "Equity Market Neutral Hedge Fund Index")
adf_test(index_data["Event Driven"], "Event Driven Hedge Fund Index")
adf_test(index_data["Fixed Income Arbitrage"], "Fixed Income Arbitrage Hedge Fund Index")
adf_test(index_data["Global Macro"], "Global Macro Hedge Fund Index")
adf_test(index_data["Merger Arbitrage"], "Merger Arbitrage Hedge Fund Index")
adf_test(index_data["Multi Strategy"], "Multi Strategy Hedge Fund Index")
adf_test(index_data["Barclay Hedge Fund Index"], "Hedge Fund Industry Index")


# I also decided to perform the ACF test to test for autocorrelation and to determine if I needed to perform first order
# differencing

# Defining the ACF function
def acf_test(series, str_name):
    plot_acf(series, lags=5, label=str_name)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(str_name + ' Autocorrelation Function (ACF)')
    plt.legend()
    plt.show()


acf_test(index_data["SPX Return"], "SPX Return")
acf_test(index_data["Convertible Arbitrage"], "Convertible Arbitrage Hedge Fund Index")
acf_test(index_data["Equity L/S"], "Equity L/S Hedge Fund Index")
acf_test(index_data["Equity Market Neutral"], "Equity Market Neutral Hedge Fund Index")
acf_test(index_data["Event Driven"], "Event Driven Hedge Fund Index")
acf_test(index_data["Fixed Income Arbitrage"], "Fixed Income Arbitrage Hedge Fund Index")
acf_test(index_data["Global Macro"], "Global Macro Hedge Fund Index")
acf_test(index_data["Merger Arbitrage"], "Merger Arbitrage Hedge Fund Index")
acf_test(index_data["Multi Strategy"], "Multi Strategy Hedge Fund Index")
acf_test(index_data["Barclay Hedge Fund Index"], "Hedge Fund Industry Index")


# I wanted to perform the PACF test to determine the AR portion of the ARMA model.
# Defining the function
def pacf_test(series, str_name):
    plot_pacf(series)
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')
    plt.title(str_name + ' Partial Autocorrelation Function (PACF)')
    plt.show()


pacf_test(index_data["SPX Return"], "SPX Return")
pacf_test(index_data["Convertible Arbitrage"], "Convertible Arbitrage Hedge Fund Index")
pacf_test(index_data["Equity L/S"], "Equity L/S Hedge Fund Index")
pacf_test(index_data["Equity Market Neutral"], "Equity Market Neutral Hedge Fund Index")
pacf_test(index_data["Event Driven"], "Event Driven Hedge Fund Index")
pacf_test(index_data["Fixed Income Arbitrage"], "Fixed Income Arbitrage Hedge Fund Index")
pacf_test(index_data["Global Macro"], "Global Macro Hedge Fund Index")
pacf_test(index_data["Merger Arbitrage"], "Merger Arbitrage Hedge Fund Index")
pacf_test(index_data["Multi Strategy"], "Multi Strategy Hedge Fund Index")
pacf_test(index_data["Barclay Hedge Fund Index"], "Hedge Fund Industry Index")


def ARIMAmodel(series, strName, int_p, differencing, int_q):

    # Plot residuals as a scatterplot (residuals vs. time)

    model = ARIMA(endog=series, exog=index_data["SPX Return"], order=(int_p, differencing, int_q)).fit()
    residuals = model.resid

    # Standardized residuals = residuals divided by their standard deviation
    standardised_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

    plt.figure(figsize=(10, 5))
    plt.scatter(index_data.index, standardised_residuals, alpha=0.6)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axhline(2, color='red', linestyle='--', linewidth=1)
    plt.axhline(-2, color='red', linestyle='--', linewidth=1)
    plt.title(strName + ' Scatterplot of Standardised Residuals')
    plt.xlabel('Time')
    plt.ylabel('Standardised Residual')
    plt.tight_layout()
    plt.show()

    return model


CA_model = ARIMAmodel(index_data["Convertible Arbitrage"], "Convertible Arbitrage Hedge Fund Index", 0, 0, 1)
Equity_LS_model = ARIMAmodel(index_data["Equity L/S"], "Equity L/S Hedge Fund Index", 0, 0, 1)
EMN_model = ARIMAmodel(index_data["Equity Market Neutral"], "Equity Market Neutral Hedge Fund Index", 1, 1, 0)
ED_model = ARIMAmodel(index_data["Event Driven"], "Event Driven Hedge Fund Index", 0, 0, 1)
FIA_model = ARIMAmodel(index_data["Fixed Income Arbitrage"], "Fixed Income Arbitrage Hedge Fund Index", 1, 0, 0)
GM_model = ARIMAmodel(index_data["Global Macro"], "Global Macro Hedge Fund Index", 0, 0, 1)
MA_model = ARIMAmodel(index_data["Merger Arbitrage"], "Merger Arbitrage Hedge Fund Index", 0, 0, 1)
MS_model = ARIMAmodel(index_data["Multi Strategy"], "Multi Strategy Hedge Fund Index", 0, 0, 1)
HF_model = ARIMAmodel(index_data["Barclay Hedge Fund Index"], "Hedge Fund Industry Index", 0, 0, 1)


# LIKELIHOOD TEST
def likelihood_test(model, series, strName, p, d, q):
    new_p = p
    new_q = q
    model_simple = model

    # if we have an AR(1), try an ARMA(1,1)
    if p > 0 and q == 0:
        new_p = 1
        new_q = 1
    elif p == 0 and q > 0:  # if we have an MA(1), try an ARMA(1,1)
        new_p = 1
        new_q = 1
    elif p > 0 and d > 0 and q == 0:  # if we have an ARIMA(1,1,0) model, try an ARIMA of (1,1,1)
        new_p = 1
        new_q = 1


    model_complex = ARIMA(exog=index_data["SPX Return"], endog=series, order=(new_p, d, new_q)).fit()

    # Calculate Log Likelihoods
    ll_simple = model_simple.llf
    ll_complex = model_complex.llf

    # Likelihood ratio statistic
    LR_stat = -2 * (ll_simple - ll_complex)

    # Degrees of freedom = difference in number of parameters
    df = model_complex.df_model - model_simple.df_model

    # p-value from chi-squared distribution
    p_value = chi2.sf(LR_stat, df)

    print(strName + " Hedge Fund Index Likelihood Ratio Test")
    print(f"Likelihood Ratio Statistic: {LR_stat:.3f}")
    print(f"Degrees of Freedom: {df}")
    print(f"P-Value: {p_value:.4f}")


likelihood_test(CA_model, index_data["Convertible Arbitrage"], "Convertible Arbitrage Hedge Fund Index", 0, 0, 1)
likelihood_test(Equity_LS_model, index_data["Equity L/S"], "Equity L/S Hedge Fund Index",  0, 0, 1)
likelihood_test(EMN_model, index_data["Equity Market Neutral"], "Equity Market Neutral Hedge Fund Index", 1, 1, 0)
likelihood_test(ED_model, index_data["Event Driven"], "Event Driven Hedge Fund Index", 0, 0, 1)
likelihood_test(FIA_model, index_data["Fixed Income Arbitrage"], "Fixed Income Arbitrage Hedge Fund Index", 1, 0, 0)
likelihood_test(GM_model, index_data["Global Macro"], "Global Macro Hedge Fund Index", 0, 0, 1)
likelihood_test(MA_model, index_data["Merger Arbitrage"], "Merger Arbitrage Hedge Fund Index", 0, 0, 1)
likelihood_test(MS_model,index_data["Multi Strategy"], "Multi Strategy Hedge Fund Index", 0, 0, 1)
likelihood_test(HF_model, index_data["Barclay Hedge Fund Index"], "Hedge Fund Industry Index", 0, 0, 1)


# Given the results of the LM Test, the more complex model for the EMN index is a much better fit
EMN_model = ARIMA(endog=index_data["Equity Market Neutral"], exog=index_data["SPX Return"], order=(1, 1, 1)).fit()

# LJUNG-BOX TEST
def ljungbox_test(model, strName):
    ljungbox_result = acorr_ljungbox(model.resid, lags=10, return_df=True)

    print(strName)
    print(ljungbox_result)


ljungbox_test(CA_model, "Convertible Arbitrage Hedge Fund Index")
ljungbox_test(Equity_LS_model, "Equity L/S Hedge Fund Index")
ljungbox_test(EMN_model, "Equity Market Neutral Hedge Fund Index")
ljungbox_test(ED_model, "Event-Driven Hedge Fund Index")
ljungbox_test(FIA_model, "Fixed Income Arbitrage Hedge Fund Index")
ljungbox_test(GM_model, "Global Macro Hedge Fund Index")
ljungbox_test(MA_model, "Merger Arbitrage Hedge Fund Index")
ljungbox_test(MS_model, "Multi Strategy Hedge Fund Index")
ljungbox_test(HF_model, "Hedge Fund Industry Index")


# BAI-PERRON TEST
def baiperron_test(series, strName):

    # Using PELT algorithm with the least squares cost
    model_type = "l2"
    algo = rpt.Pelt(model=model_type).fit(series.values)
    breaks = algo.predict(pen=3)
    print("Breakpoints at times: ", breaks)
    rpt.display(series.values, breaks)
    plt.title("Detected Structural Breaks for " + strName + " Hedge Fund Index Returns Data")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.show()



# F-STATISTIC COMPUTATION FOR BAI-PERRON TEST
def compute_f_stat(model, strName, x, y, break_index, p, d, q, alpha = 0.05):
    # Full model
    full_model = model
    SSR_full = np.sum(full_model.resid ** 2)
    k = len(full_model.params)


    # Segment 1
    y1, X1 = y.iloc[:break_index], x.iloc[:break_index]
    model1 = ARIMA(endog=y1, exog=X1, order=(p, d, q)).fit()
    SSR1 = np.sum(model1.resid ** 2)


    # Segment 2
    y2, X2 = y.iloc[break_index:], x.iloc[break_index:]
    model2 = ARIMA(endog=y2, exog=X2, order=(p, d, q)).fit()
    SSR2 = np.sum(model2.resid ** 2)

    T = len(y)
    numerator = (SSR_full - (SSR1 + SSR2)) / k
    denominator = (SSR1 + SSR2) / (T - 2 * k)
    F_stat = numerator / denominator

    # Critical value
    df1, df2 = k, T - 2 * k
    F_crit = f.ppf(1 - alpha, df1, df2)
    significant = F_stat > F_crit

    print(strName)
    print("Break index: ", break_index)
    print("F-Stat: ", F_stat)
    print("F-Critical Value: ", F_crit)
    print("p-value: ", 1-f.cdf(F_stat,df1,df2))
    print("significant: ", significant)


baiperron_test(index_data["SPX Return"], "SPX Return")
baiperron_test(index_data["Convertible Arbitrage"], "Convertible Arbitrage")
baiperron_test(index_data["Equity L/S"], "Equity L/S")
baiperron_test(index_data["Equity Market Neutral"], "Equity Market Neutral")
baiperron_test(index_data["Event Driven"], "Event Driven")
baiperron_test(index_data["Fixed Income Arbitrage"], "Fixed Income Arbitage")
baiperron_test(index_data["Global Macro"], "Global Macro")
baiperron_test(index_data["Merger Arbitrage"], "Merger Arbitrage")
baiperron_test(index_data["Multi Strategy"], "Multi Strategy")
baiperron_test(index_data["Barclay Hedge Fund Index"], "Barclay Hedge Fund Index")


# FINAL MODEL SPECIFICATIONS
print("Convertible Arbitrage")
print(CA_model.params)
print("Equity L/S")
print(Equity_LS_model.params)
print("EMN")
print(EMN_model.params)
print("E-D")
print(ED_model.params)
print("FIA")
print(FIA_model.params)
print("GM")
print(GM_model.params)
print("MA")
print(MA_model.params)
print("M-S")
print(MS_model.params)
print("HF Industry")
print(HF_model.params)


# HYPOTHESIS TESTING

MS_averages = [0.386, 0.796, -0.243, 0.010]
CA_averages = [1.247, 0.581, -0.108, 0.298]
EquityLS_averages = [0.785, 0.863, 0.002, 0.250]
EMN_averages = [-0.106, 0.648, 0.245, 0.294]
ED_averages = [1.009, 0.983, -0.516, 0.042]
FIA_averages = [0.837, 0.083, -0.138, 0.330]
GM_averages = [0.825, 0.715, 0.541, -0.140]
MA_averages = [0.784, 0.627, 0.083, -0.168]
HFIndex_averages = [0.961, 0.821, -0.688, 0.420]


def difference_of_means(x_averages, y_averages, name):
    # for i in range(4):
    #
    #     diff = x_averages - y_averages
    #     # Paired t-test
    #     t_stat, p_val_t = stats.ttest_rel(x_averages[i], y_averages[i])
    #
    #     if diff.std(ddof=1) == 0:
    #         print(f"All differences are identical (mean diff = {diff.mean():.6f}).")
    #         print("Paired t-test is not defined (zero variance).")
    #     else:
    #         print(name + " Paired t-test:")
    #         print(f"  t-stat = {t_stat:.4f}, p-value = {p_val_t:.4f}")

    t_stat, p_val_t = stats.ttest_rel(x_averages, y_averages)
    print(name + " Paired t-test:")
    print(f"  t-stat = {t_stat:.4f}, p-value = {p_val_t:.4f}")



difference_of_means(MS_averages, CA_averages, "Multi-Strategy vs. Convertible Arbitrage")
difference_of_means(MS_averages, EquityLS_averages, "Multi-Strategy vs. Equity L/S")
difference_of_means(MS_averages, EMN_averages, "Multi-Strategy vs. Equity Market Neutral")
difference_of_means(MS_averages, ED_averages, "Multi-Strategy vs. Event-Driven")
difference_of_means(MS_averages, FIA_averages, "Multi-Strategy vs. Fixed Income Arbitrage")
difference_of_means(MS_averages, GM_averages, "Multi-Strategy vs. Global Macro")
difference_of_means(MS_averages, MA_averages, "Multi-Strategy vs. Merger Arbitrage")
difference_of_means(MS_averages, HFIndex_averages, "Multi-Strategy vs. BarclayHedge Fund Index")



# FORECASTING CALCULATIONS
def MA_forecast(data_set, spx_data, name, beta, mu, theta):
    epsilons = []

    # HISTORICAL EPSILON CALCULATIONS
    for i in range(90):
        calculated_epsilon = 0
        previous_epsilon = 0

        if i == 0:
            previous_epsilon = 0
        else:
            previous_epsilon = epsilons[i - 1]

        calculated_epsilon = data_set.iloc[i] - (beta * spx_data.iloc[i]) - mu - (theta * previous_epsilon)

        epsilons.append(calculated_epsilon)

    # JULY FORECAST
    estimated_y = (beta * 0.021667) + mu + 0 + (theta * epsilons[len(epsilons)-1])
    print(name + "'s JULY FORECAST: " + str(estimated_y))



def AR_forecast(data_set, spx_data, name, beta, mu, phi,):
    epsilons = []

    # HISTORICAL EPSILON CALCULATIONS
    for i in range(90):
        calculated_epsilon = 0
        previous_epsilon = 0

        if i == 0:
            previous_epsilon = 0
        else:
            previous_epsilon = epsilons[i - 1]

        calculated_epsilon = data_set.iloc[i] - (beta * spx_data.iloc[i]) - mu - (phi * data_set.iloc[i-1])

        epsilons.append(calculated_epsilon)

    # JULY FORECAST
    estimated_y = (beta * 0.021667) + mu + (phi * data_set.iloc[len(data_set) - 1]) + 0
    print(name + "'s JULY FORECAST: " + str(estimated_y))


def ARMA_forecast(data_set, spx_data, name, beta, phi, theta):
    epsilons = []

    # HISTORICAL EPSILON CALCULATIONS
    for i in range(90):
        calculated_epsilon = 0
        previous_epsilon = 0

        if i == 0:
            previous_epsilon = 0
        else:
            previous_epsilon = epsilons[i - 1]
        if i > 1:
            calculated_epsilon = data_set.iloc[i] - (beta * spx_data.iloc[i]) - (phi * (data_set.iloc[i - 1] - data_set.iloc[i - 2])) - (theta * epsilons[len(epsilons) - 1])

        epsilons.append(calculated_epsilon)

    # JULY FORECAST
    estimated_y = (beta * 0.021667) + (phi * (data_set.iloc[len(data_set)- 1] - data_set.iloc[len(data_set) - 2])) + (theta * epsilons[len(epsilons) - 1])
    print(name + "'s JULY FORECAST: " + str(estimated_y))


MA_forecast(index_data["Convertible Arbitrage"], index_data["SPX Return"], "Convertible Arbitrage",0.139233, 0.004160, 0.135369)
MA_forecast(index_data["Equity L/S"], index_data["SPX Return"], "Equity L/S",0.236798, 0.002883, 0.051052)
ARMA_forecast(index_data["Equity Market Neutral"], index_data["SPX Return"], "Equity Market Neutral", 0.161284,0.031837, -0.847112)
MA_forecast(index_data["Event Driven"], index_data["SPX Return"], "Event-Driven Arbitrage", 0.385806, 0.001628, 0.075582)
AR_forecast(index_data["Fixed Income Arbitrage"], index_data["SPX Return"], "Fixed Income Arbitrage", 0.039215,0.003579, 0.263777)
MA_forecast(index_data["Global Macro"], index_data["SPX Return"], "Global Macro", 0.184167,0.003516,0.042511)
MA_forecast(index_data["Merger Arbitrage"], index_data["SPX Return"], "Merger Arbitrage", 0.161284, 0.002766, -0.020043)
MA_forecast(index_data["Multi Strategy"], index_data["SPX Return"], "Multi-Strategy", 0.216068,0.000897,0.090294)
MA_forecast(index_data["Barclay Hedge Fund Index"], index_data["SPX Return"], "BarclayHedge Fund Index", 0.361215, 0.001652, -0.133401)

# END
