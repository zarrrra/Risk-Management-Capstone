import io, math, json, time, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title = 'PriceOnlyRiskDashboardGoBlue', layout = 'wide')
st.markdown('''
<style>
    .stApp { background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%); }
    h1 { color: #00274C !important; font-family: Arial Black, sans-serif; text-transform: uppercase; letter-spacing: 2px; border-bottom: 4px solid #FFCB05; padding-bottom: 10px; text-shadow: 2px 2px 4px rgba(0, 39, 76, 0.1); }
    [data-testid="metric-container"] { background: linear-gradient(135deg, #00274C 0%, #003366 100%); padding: 20px; border-radius: 10px; border: 2px solid #FFCB05; box-shadow: 0 6px 12px rgba(0, 39, 76, 0.2); }
    [data-testid="metric-container"] label { color: #FFCB05 !important; font-weight: 700; text-transform: uppercase; font-size: 12px !important; letter-spacing: 1px; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color: white !important; font-size: 28px !important; font-weight: bold; }
    [data-testid="stSidebar"] { background: #00274C; }
    [data-testid="stSidebar"] h2 { color: #FFCB05 !important; }
    [data-testid="stSidebar"] label { color: white !important; }
    .stButton > button { background: linear-gradient(135deg, #FFCB05 0%, #FFD733 100%); color: #00274C; font-weight: bold; border: none; border-radius: 5px; padding: 10px 20px; text-transform: uppercase; letter-spacing: 1px; transition: all 0.3s ease; }
    .stButton > button:hover { background: linear-gradient(135deg, #FFD733 0%, #FFCB05 100%); box-shadow: 0 4px 8px rgba(255, 203, 5, 0.3); transform: translateY(-2px); }
    .stAlert { background: rgba(0, 39, 76, 0.05); border: 1px solid #00274C; border-left: 5px solid #FFCB05; }
</style>
''', unsafe_allow_html = True)
st.markdown('''
<div style='text-align: center; margin-bottom: 30px;'>
  <div style='display: inline-block; padding: 20px 40px; background: #00274C; border-radius: 10px; border: 3px solid #FFCB05;'>
    <h2 style='color: #FFCB05; margin: 0; font-family: Arial Black; letter-spacing: 3px;'>RISK ANALYTICS DASHBOARD</h2>
    <p style='color: white; margin: 5px 0 0 0; font-size: 14px; letter-spacing: 2px;'>Master's Degree of Applied Data Science | Capstone Research Project</p>
  </div>
</div>
''', unsafe_allow_html = True)
def _single_page_tabs(labels):
    return [st.container() for _ in labels]
st.tabs = _single_page_tabs


def fetchAlphaVantage(symbol: str, apiKey: str, years: int = 3) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    parameters = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": apiKey
    }
    response = requests.get(url, params = parameters, timeout = 30)
    data = response.json()
    timeSeriesKey = "Time Series (Daily)"

    if timeSeriesKey not in data:
        message = data.get("Note") or data.get("Error Message") or "Unknown error"
        raise RuntimeError(message)

    frame = pd.DataFrame.from_dict(data[timeSeriesKey], orient = "index").apply(pd.to_numeric, errors = "coerce")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    renameMap = {
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    }
    frame = frame.rename(columns = renameMap)

    cutoffDate = datetime.now() - timedelta(days = 365 * years + 5)
    frame = frame[frame.index >= cutoffDate]

    keepColumns = ["Open", "High", "Low", "Close", "Volume"]
    existingColumns = [column for column in keepColumns if column in frame.columns]
    frame = frame[existingColumns].dropna(how = "any")

    if frame.empty:
        raise RuntimeError("No recent data returned for " + symbol + " after trimming to " + str(years) + " years.")

    return frame


def computeReturns(priceSeries: pd.Series, useLog: bool = True) -> pd.Series:
    if useLog:
        returnSeries = np.log(priceSeries).diff()
    else:
        returnSeries = priceSeries.pct_change()
    return returnSeries.dropna()


def fitStudentT(returnSeries: pd.Series):
    degreesFreedom, locationValue, scaleValue = stats.t.fit(returnSeries.values)
    return degreesFreedom, locationValue, scaleValue


def buildTPdfOverlay(
    returnSeries: pd.Series,
    degreesFreedom: float,
    locationValue: float,
    scaleValue: float,
    binCount: int = 50
):
    histogramHeights, histogramEdges = np.histogram(returnSeries, bins = binCount, density = True)
    histogramMidpoints = 0.5 * (histogramEdges[1:] + histogramEdges[:-1])
    evaluationGrid = np.linspace(histogramMidpoints.min(), histogramMidpoints.max(), 400)
    pdfValues = stats.t.pdf((evaluationGrid - locationValue) / scaleValue, degreesFreedom) / scaleValue
    return histogramMidpoints, histogramHeights, evaluationGrid, pdfValues


def computeVarCvar(alphaLevel: float, degreesFreedom: float, locationValue: float, scaleValue: float):
    quantileValue = stats.t.ppf(alphaLevel, degreesFreedom, loc = locationValue, scale = scaleValue)
    evaluationGrid = np.linspace(quantileValue - 10 * scaleValue, quantileValue, 4000)
    pdfValues = stats.t.pdf((evaluationGrid - locationValue) / scaleValue, degreesFreedom) / scaleValue
    expectedShortfall = np.trapz(evaluationGrid * pdfValues, evaluationGrid) / alphaLevel
    return quantileValue, expectedShortfall


def computeSharpeApproximation(returnSeries: pd.Series, tradingDays: int = 252):
    meanDaily = returnSeries.mean()
    deviationDaily = returnSeries.std(ddof = 1)
    meanAnnual = meanDaily * tradingDays
    deviationAnnual = deviationDaily * math.sqrt(tradingDays)

    if deviationAnnual == 0:
        sharpeValue = 0.0
    else:
        sharpeValue = meanAnnual / deviationAnnual

    return meanDaily, deviationDaily, meanAnnual, deviationAnnual, sharpeValue


def computeRollingVolatility(
    returnSeries: pd.Series,
    windowSize: int = 30,
    tradingDays: int = 252
) -> pd.Series:
    return returnSeries.rolling(windowSize).std(ddof = 1) * math.sqrt(tradingDays)


def computeDownsideDeviation(
    returnSeries: pd.Series,
    threshold: float = 0,
    tradingDays: int = 252
) -> float:
    downsideReturns = returnSeries[returnSeries < threshold]
    if len(downsideReturns) < 2:
        return 0.0
    downsideStd = downsideReturns.std(ddof = 1)
    return downsideStd * math.sqrt(tradingDays)


def computeSortinoRatio(returnSeries: pd.Series, tradingDays: int = 252) -> float:
    meanReturn = returnSeries.mean() * tradingDays
    downsideDeviation = computeDownsideDeviation(returnSeries, 0, tradingDays)
    if downsideDeviation == 0:
        return 0.0
    return meanReturn / downsideDeviation


def computeCalmarRatio(
    returnSeries: pd.Series,
    maxDrawdown: float,
    tradingDays: int = 252
) -> float:
    annualReturn = returnSeries.mean() * tradingDays
    if maxDrawdown == 0:
        return 0.0
    return abs(annualReturn / maxDrawdown)


def computeUlcerIndex(drawdownSeries: pd.Series) -> float:
    squaredDrawdowns = drawdownSeries ** 2
    ulcerIndex = math.sqrt(squaredDrawdowns.mean())
    return ulcerIndex


def computeRecoveryMetrics(priceSeries: pd.Series):
    runningMax = priceSeries.cummax()
    drawdowns = (priceSeries - runningMax) / runningMax

    inDrawdown = drawdowns < 0
    recoveryPeriods = []
    currentPeriod = 0

    for indexValue in range(1, len(inDrawdown)):
        if inDrawdown.iloc[indexValue]:
            currentPeriod += 1
        else:
            if currentPeriod > 0 and not inDrawdown.iloc[indexValue - 1]:
                recoveryPeriods.append(currentPeriod)
                currentPeriod = 0

    avgRecovery = np.mean(recoveryPeriods) if recoveryPeriods else 0
    maxRecovery = max(recoveryPeriods) if recoveryPeriods else 0

    return avgRecovery, maxRecovery


def computeTailRatios(returnSeries: pd.Series):
    sortedReturns = np.sort(returnSeries)
    n = len(sortedReturns)

    if n == 0:
        return 0.0, 0.0, 0.0

    bottomDecile = sortedReturns[: int(n * 0.1)].mean() if n >= 10 else sortedReturns.mean()
    topDecile = sortedReturns[-int(n * 0.1):].mean() if n >= 10 else sortedReturns.mean()

    gainLossRatio = abs(topDecile / bottomDecile) if bottomDecile != 0 else 0

    positiveReturns = returnSeries[returnSeries > 0]
    negativeReturns = returnSeries[returnSeries < 0]

    hitRate = len(positiveReturns) / len(returnSeries) if len(returnSeries) > 0 else 0
    avgWin = positiveReturns.mean() if len(positiveReturns) > 0 else 0
    avgLoss = negativeReturns.mean() if len(negativeReturns) > 0 else 0

    profitFactor = abs(avgWin / avgLoss) if avgLoss != 0 else 0

    return gainLossRatio, hitRate, profitFactor


def computeRollingBeta(assetReturns: pd.Series, marketReturns: pd.Series, window: int = 60) -> pd.Series:
    rollingCov = assetReturns.rolling(window).cov(marketReturns)
    rollingVar = marketReturns.rolling(window).var()
    rollingBeta = rollingCov / rollingVar
    return rollingBeta


def computeDrawdownStats(priceSeries: pd.Series):
    runningPeaks = priceSeries.cummax()
    drawdownSeries = priceSeries / runningPeaks - 1.0
    maximumDrawdown = drawdownSeries.min()
    currentDrawdown = drawdownSeries.iloc[-1]

    isDrawdown = drawdownSeries < 0
    durations = []
    currentDuration = 0

    for isInDrawdown in isDrawdown.values:
        if isInDrawdown:
            currentDuration = currentDuration + 1
        else:
            if currentDuration > 0:
                durations.append(currentDuration)
            currentDuration = 0

    if currentDuration > 0:
        durations.append(currentDuration)

    if durations:
        longestDuration = int(max(durations))
    else:
        longestDuration = 0

    return drawdownSeries, float(maximumDrawdown), float(currentDrawdown), longestDuration


def labelVolatilityRegime(
    rollingVolatilitySeries: pd.Series,
    lowQuantile: float = 0.33,
    highQuantile: float = 0.66
):
    cleanedSeries = rollingVolatilitySeries.dropna()
    if cleanedSeries.empty:
        return "NotAvailable", float("nan"), float("nan")

    latestValue = cleanedSeries.iloc[-1]
    lowThreshold = cleanedSeries.quantile(lowQuantile)
    highThreshold = cleanedSeries.quantile(highQuantile)

    if latestValue <= lowThreshold:
        regimeLabel = "LowVol"
    elif latestValue >= highThreshold:
        regimeLabel = "HighVol"
    else:
        regimeLabel = "MediumVol"

    return regimeLabel, float(lowThreshold), float(highThreshold)


def labelTrend(priceSeries: pd.Series, fastWindow: int = 50, slowWindow: int = 200):
    movingAverageFast = priceSeries.rolling(fastWindow).mean()
    movingAverageSlow = priceSeries.rolling(slowWindow).mean()

    if math.isnan(movingAverageFast.iloc[-1]) or math.isnan(movingAverageSlow.iloc[-1]):
        return "NotAvailable", movingAverageFast, movingAverageSlow

    if movingAverageFast.iloc[-1] > movingAverageSlow.iloc[-1]:
        return "UptrendMA" + str(fastWindow) + "Above" + str(slowWindow), movingAverageFast, movingAverageSlow

    if movingAverageFast.iloc[-1] < movingAverageSlow.iloc[-1]:
        return "DowntrendMA" + str(fastWindow) + "Below" + str(slowWindow), movingAverageFast, movingAverageSlow

    return "NeutralTrend", movingAverageFast, movingAverageSlow


def simulateStudentTPaths(
    startPrice: float,
    dayCount: int,
    pathCount: int,
    degreesFreedom: float,
    locationValue: float,
    scaleValue: float,
    seedValue: int = 42
) -> np.ndarray:
    randomGenerator = np.random.default_rng(seedValue)
    returnArray = stats.t.rvs(
        degreesFreedom,
        loc = locationValue,
        scale = scaleValue,
        size = (dayCount, pathCount),
        random_state = randomGenerator
    )
    pathArray = np.empty_like(returnArray)

    pathArray[0, :] = startPrice * np.exp(returnArray[0, :])

    for timeIndex in range(1, dayCount):
        pathArray[timeIndex, :] = pathArray[timeIndex - 1, :] * np.exp(returnArray[timeIndex, :])

    return pathArray


def computeRollingSharpeSeries(
    returnSeries: pd.Series,
    windowSize: int = 63,
    tradingDays: int = 252
) -> pd.Series:
    """
    Rolling Sharpe series (useful as a time-series feature for clustering later).
    """
    rollingMean = returnSeries.rolling(windowSize).mean()
    rollingStd = returnSeries.rolling(windowSize).std(ddof = 1)
    rollingSharpe = (rollingMean / rollingStd) * math.sqrt(tradingDays)
    rollingSharpe = rollingSharpe.replace([np.inf, -np.inf], np.nan)
    return rollingSharpe


def computeMonthlyReturnFrame(priceSeries: pd.Series) -> pd.DataFrame:
    """
    Monthly return frame that can be used for time-period clustering
    and calendar-style visuals.
    """
    monthlyPrices = priceSeries.resample("M").last()
    monthlyReturns = monthlyPrices.pct_change().dropna()

    monthlyFrame = monthlyReturns.to_frame(name = "MonthlyReturn")
    monthlyFrame["Year"] = monthlyFrame.index.year
    monthlyFrame["Month"] = monthlyFrame.index.month
    monthlyFrame["YearMonthLabel"] = monthlyFrame.index.strftime("%Y-%m")

    return monthlyFrame


def readSidebarInputs():
    with st.sidebar:
        st.markdown(
            "<h2 style='color: #00274C; border-bottom: 2px solid #FFCB05; padding-bottom: 10px;'>Configuration</h2>",
            unsafe_allow_html = True
        )

        symbolInput = st.text_input("TickerSymbol", value = "AAPL").strip().upper()
        yearsInput = st.slider("HistoricalYears", 2, 10, 3, 1)
        rollingWindowInput = st.slider("RollingVolatilityWindowDays", 10, 90, 30, 5)
        movingAverageFastWindow = st.slider("FastMovingAverageDays", 10, 100, 50, 5)
        movingAverageSlowWindow = st.slider("SlowMovingAverageDays", 100, 300, 200, 10)
        alphaChoice = st.selectbox("VarCvarTailAlpha", options = [0.01, 0.025, 0.05], index = 2)
        visiblePathCount = st.slider("MonteCarloPathsDisplay", 5, 100, 25, 5)
        histogramPathCount = st.select_slider(
            "MonteCarloSimulations",
            options = [10000, 20000, 50000, 100000],
            value = 50000
        )
        simulationDayCount = st.select_slider(
            "SimulationHorizonDays",
            options = [126, 189, 252],
            value = 252
        )
        seedInput = st.number_input("RandomSeed", value = 42, step = 1)
        apiKeyInput = st.text_input(
            "AlphaVantageApiKey",
            value = os.getenv("ALPHAVANTAGE_API_KEY", ""),
            type = "password"
        )
        st.markdown("<br>", unsafe_allow_html = True)
        runButton = st.button("Analyze", use_container_width = True)

    return {
        "symbol": symbolInput,
        "years": yearsInput,
        "rollingWindow": rollingWindowInput,
        "fastWindow": movingAverageFastWindow,
        "slowWindow": movingAverageSlowWindow,
        "alpha": alphaChoice,
        "visiblePaths": visiblePathCount,
        "histogramPaths": histogramPathCount,
        "simulationDays": simulationDayCount,
        "seed": seedInput,
        "apiKeyInput": apiKeyInput,
        "run": runButton
    }


def loadPriceSeries(inputParameters):
    if not inputParameters["run"]:
        st.info("Set parameters in the sidebar and select Analyze.")
        return None

    apiKey = inputParameters["apiKeyInput"] or os.getenv("ALPHAVANTAGE_API_KEY")

    if not apiKey:
        st.error("Alpha Vantage api key is required.")
        return None

    try:
        with st.spinner(
            "Loading " + str(inputParameters["years"]) + " years of data for " + inputParameters["symbol"] + "..."
        ):
            priceFrame = fetchAlphaVantage(
                inputParameters["symbol"],
                apiKey,
                years = inputParameters["years"]
            )
    except Exception as exceptionValue:
        st.error("Error fetching data for " + inputParameters["symbol"] + ": " + str(exceptionValue))
        return None

    priceSeries = priceFrame["Close"].copy()
    priceSeries.index = pd.to_datetime(priceSeries.index)

    if getattr(priceSeries.index, "tz", None) is not None:
        priceSeries = priceSeries.tz_localize(None)

    return priceSeries


def computeAllMetrics(priceSeries, inputParameters):
    returnSeries = computeReturns(priceSeries, useLog = True)

    degreesFreedom, locationValue, scaleValue = fitStudentT(returnSeries)
    meanDaily, deviationDaily, meanAnnual, deviationAnnual, sharpeValue = computeSharpeApproximation(
        returnSeries,
        tradingDays = 252
    )
    varValue, cvarValue = computeVarCvar(
        inputParameters["alpha"],
        degreesFreedom,
        locationValue,
        scaleValue
    )

    rollingVolatilitySeries = computeRollingVolatility(
        returnSeries,
        windowSize = inputParameters["rollingWindow"],
        tradingDays = 252
    )
    volatilityRegimeLabel, lowVolatilityThreshold, highVolatilityThreshold = labelVolatilityRegime(
        rollingVolatilitySeries
    )

    drawdownSeries, maximumDrawdown, currentDrawdown, drawdownDays = computeDrawdownStats(priceSeries)
    trendLabelValue, movingAverageFast, movingAverageSlow = labelTrend(
        priceSeries,
        fastWindow = inputParameters["fastWindow"],
        slowWindow = inputParameters["slowWindow"]
    )

    downsideDeviation = computeDownsideDeviation(returnSeries)
    sortinoRatio = computeSortinoRatio(returnSeries)
    calmarRatio = computeCalmarRatio(returnSeries, maximumDrawdown)
    ulcerIndex = computeUlcerIndex(drawdownSeries)

    avgRecoveryTime, maxRecoveryTime = computeRecoveryMetrics(priceSeries)
    gainLossRatio, hitRate, profitFactor = computeTailRatios(returnSeries)

    rollingSharpeShortSeries = computeRollingSharpeSeries(
        returnSeries,
        windowSize = 63,
        tradingDays = 252
    )
    rollingSharpeLongSeries = computeRollingSharpeSeries(
        returnSeries,
        windowSize = 126,
        tradingDays = 252
    )

    rollingWindow = inputParameters["rollingWindow"]
    rollingMeanReturnSeries = returnSeries.rolling(rollingWindow).mean() * 252

    rollingDownsideDeviationSeries = returnSeries.rolling(rollingWindow).apply(
        lambda windowValues: computeDownsideDeviation(pd.Series(windowValues), tradingDays = 252),
        raw = False
    )
    rollingDownsideDeviationSeries = rollingDownsideDeviationSeries.replace([np.inf, -np.inf], np.nan)

    rollingReturnQuantileLowSeries = returnSeries.rolling(rollingWindow).quantile(0.05)
    rollingReturnQuantileHighSeries = returnSeries.rolling(rollingWindow).quantile(0.95)

    varBreachSeries = (returnSeries < varValue).astype(int)
    varBreachRollingCountSeries = varBreachSeries.rolling(rollingWindow).sum()

    rollingSortinoSeries = pd.Series(index = rollingMeanReturnSeries.index, dtype = float)
    validSortinoMask = (
        rollingDownsideDeviationSeries.notna()
        & (rollingDownsideDeviationSeries > 0)
        & rollingMeanReturnSeries.notna()
    )
    rollingSortinoSeries[validSortinoMask] = (
        rollingMeanReturnSeries[validSortinoMask] / rollingDownsideDeviationSeries[validSortinoMask]
    )

    monthlyReturnFrame = computeMonthlyReturnFrame(priceSeries)
    if monthlyReturnFrame is not None and not monthlyReturnFrame.empty:
        avgMonthlyReturn = float(monthlyReturnFrame["MonthlyReturn"].mean())
    else:
        avgMonthlyReturn = float("nan")

    volatilityRegimeSeries = None
    if not rollingVolatilitySeries.dropna().empty and not math.isnan(lowVolatilityThreshold) and not math.isnan(highVolatilityThreshold):
        volRegime = pd.Series(index = rollingVolatilitySeries.index, dtype = float)
        volRegime[rollingVolatilitySeries <= lowVolatilityThreshold] = 0.0
        volRegime[(rollingVolatilitySeries > lowVolatilityThreshold) & (rollingVolatilitySeries < highVolatilityThreshold)] = 1.0
        volRegime[rollingVolatilitySeries >= highVolatilityThreshold] = 2.0
        volatilityRegimeSeries = volRegime

    returnRegimeSeries = None
    if rollingMeanReturnSeries is not None and rollingMeanReturnSeries.notna().sum() > 0:
        reg = pd.Series(index = rollingMeanReturnSeries.index, dtype = float)
        reg[rollingMeanReturnSeries > 0] = 1.0
        reg[rollingMeanReturnSeries < 0] = -1.0
        reg[rollingMeanReturnSeries == 0] = 0.0
        returnRegimeSeries = reg

    drawdownEventsFrame = None
    drawdownEvents = []
    inEvent = False
    currentDepth = 0.0
    currentDuration = 0
    lastDate = None

    for timestamp, ddValue in drawdownSeries.items():
        if ddValue < 0:
            if not inEvent:
                inEvent = True
                currentDepth = ddValue
                currentDuration = 1
            else:
                currentDuration += 1
                if ddValue < currentDepth:
                    currentDepth = ddValue
            lastDate = timestamp
        else:
            if inEvent:
                drawdownEvents.append(
                    {
                        "Depth": float(currentDepth),
                        "Duration": int(currentDuration),
                        "EndDate": lastDate
                    }
                )
                inEvent = False
                currentDepth = 0.0
                currentDuration = 0
                lastDate = None

    if inEvent and lastDate is not None:
        drawdownEvents.append(
            {
                "Depth": float(currentDepth),
                "Duration": int(currentDuration),
                "EndDate": lastDate
            }
        )

    if len(drawdownEvents) > 0:
        drawdownEventsFrame = pd.DataFrame(drawdownEvents)

    simulationPathsVisible = simulateStudentTPaths(
        priceSeries.iloc[-1],
        dayCount = inputParameters["simulationDays"],
        pathCount = inputParameters["visiblePaths"],
        degreesFreedom = degreesFreedom,
        locationValue = locationValue,
        scaleValue = scaleValue,
        seedValue = inputParameters["seed"]
    )
    simulationFinalsArray = simulateStudentTPaths(
        priceSeries.iloc[-1],
        dayCount = inputParameters["simulationDays"],
        pathCount = inputParameters["histogramPaths"],
        degreesFreedom = degreesFreedom,
        locationValue = locationValue,
        scaleValue = scaleValue,
        seedValue = inputParameters["seed"] + 1
    )[-1, :]

    percentileLow, percentileHigh = np.percentile(simulationFinalsArray, [1.25, 98.75])
    trimmedFinals = simulationFinalsArray[
        (simulationFinalsArray >= percentileLow) & (simulationFinalsArray <= percentileHigh)
    ]

    histogramMidpoints, histogramHeights, evaluationGrid, pdfValues = buildTPdfOverlay(
        returnSeries,
        degreesFreedom,
        locationValue,
        scaleValue,
        binCount = 50
    )

    return {
        "returnSeries": returnSeries,
        "degreesFreedom": degreesFreedom,
        "locationValue": locationValue,
        "scaleValue": scaleValue,
        "meanDaily": meanDaily,
        "deviationDaily": deviationDaily,
        "meanAnnual": meanAnnual,
        "deviationAnnual": deviationAnnual,
        "sharpeValue": sharpeValue,
        "varValue": varValue,
        "cvarValue": cvarValue,
        "rollingVolatilitySeries": rollingVolatilitySeries,
        "volatilityRegimeLabel": volatilityRegimeLabel,
        "lowVolatilityThreshold": lowVolatilityThreshold,
        "highVolatilityThreshold": highVolatilityThreshold,
        "drawdownSeries": drawdownSeries,
        "maximumDrawdown": maximumDrawdown,
        "currentDrawdown": currentDrawdown,
        "drawdownDays": drawdownDays,
        "trendLabel": trendLabelValue,
        "movingAverageFast": movingAverageFast,
        "movingAverageSlow": movingAverageSlow,
        "simulationPathsVisible": simulationPathsVisible,
        "simulationFinalsArray": simulationFinalsArray,
        "trimmedFinals": trimmedFinals,
        "histogramMidpoints": histogramMidpoints,
        "histogramHeights": histogramHeights,
        "evaluationGrid": evaluationGrid,
        "pdfValues": pdfValues,
        "downsideDeviation": downsideDeviation,
        "sortinoRatio": sortinoRatio,
        "calmarRatio": calmarRatio,
        "ulcerIndex": ulcerIndex,
        "avgRecoveryTime": avgRecoveryTime,
        "maxRecoveryTime": maxRecoveryTime,
        "gainLossRatio": gainLossRatio,
        "hitRate": hitRate,
        "profitFactor": profitFactor,
        "rollingSharpeShortSeries": rollingSharpeShortSeries,
        "rollingSharpeLongSeries": rollingSharpeLongSeries,
        "monthlyReturnFrame": monthlyReturnFrame,
        "avgMonthlyReturn": avgMonthlyReturn,
        "rollingMeanReturnSeries": rollingMeanReturnSeries,
        "rollingDownsideDeviationSeries": rollingDownsideDeviationSeries,
        "volatilityRegimeSeries": volatilityRegimeSeries,
        "rollingReturnQuantileLowSeries": rollingReturnQuantileLowSeries,
        "rollingReturnQuantileHighSeries": rollingReturnQuantileHighSeries,
        "varBreachRollingCountSeries": varBreachRollingCountSeries,
        "rollingSortinoSeries": rollingSortinoSeries,
        "returnRegimeSeries": returnRegimeSeries,
        "drawdownEventsFrame": drawdownEventsFrame
    }


def renderDashboard(priceSeries, inputParameters, metrics):
    st.title("Risk Plotting | for MADS Capstone")

    umichLayout = dict(
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        font = dict(color = "#00274C", family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        title_font = dict(size = 18, color = "#00274C", family = "sans-serif"),
        showlegend = True,
        hovermode = "x unified",
        xaxis = dict(
            gridcolor = "rgba(225, 228, 232, 0.5)",
            showgrid = True,
            zeroline = False,
            linecolor = "#e1e4e8"
        ),
        yaxis = dict(
            gridcolor = "rgba(225, 228, 232, 0.5)",
            showgrid = True,
            zeroline = False,
            linecolor = "#e1e4e8"
        ),
        margin = dict(t = 50, l = 0, r = 0, b = 0)
    )

    st.subheader("Performance Overview")

    metricColumnOne, metricColumnTwo, metricColumnThree, metricColumnFour = st.columns(4)
    metricColumnOne.metric("CurrentPrice", "$" + format(priceSeries.iloc[-1], ",.2f"))
    metricColumnTwo.metric("AnnualReturn", "{:.2f}%".format(100 * metrics["meanAnnual"]))
    metricColumnThree.metric("AnnualVolatility", "{:.2f}%".format(100 * metrics["deviationAnnual"]))
    metricColumnFour.metric("SharpeRatio", "{:.2f}".format(metrics["sharpeValue"]))

    metricColumnFive, metricColumnSix, metricColumnSeven, metricColumnEight = st.columns(4)
    metricColumnFive.metric("SortinoRatio", "{:.2f}".format(metrics["sortinoRatio"]))
    metricColumnSix.metric("CalmarRatio", "{:.2f}".format(metrics["calmarRatio"]))
    metricColumnSeven.metric("HitRate", "{:.1f}%".format(100 * metrics["hitRate"]))
    metricColumnEight.metric("ProfitFactor", "{:.2f}".format(metrics["profitFactor"]))

    metricColumnNine, metricColumnTen, metricColumnEleven, metricColumnTwelve = st.columns(4)
    latestRollingSharpeShort = metrics["rollingSharpeShortSeries"].dropna().iloc[-1] if metrics["rollingSharpeShortSeries"].notna().sum() > 0 else float("nan")
    latestRollingSharpeLong = metrics["rollingSharpeLongSeries"].dropna().iloc[-1] if metrics["rollingSharpeLongSeries"].notna().sum() > 0 else float("nan")
    metricColumnNine.metric("RollingSharpe63", "{:.2f}".format(latestRollingSharpeShort) if not math.isnan(latestRollingSharpeShort) else "N/A")
    metricColumnTen.metric("RollingSharpe126", "{:.2f}".format(latestRollingSharpeLong) if not math.isnan(latestRollingSharpeLong) else "N/A")
    metricColumnEleven.metric("AvgMonthlyReturn", "{:.2f}%".format(100 * metrics["avgMonthlyReturn"]) if not math.isnan(metrics["avgMonthlyReturn"]) else "N/A")
    metricColumnTwelve.metric("DrawdownDays", str(metrics["drawdownDays"]) + " days")

    priceFigure = go.Figure()
    priceFigure.add_trace(
        go.Scatter(
            x = priceSeries.index,
            y = priceSeries,
            name = "ClosePrice",
            mode = "lines",
            line = dict(color = "#00274C", width = 2)
        )
    )
    if metrics["movingAverageFast"].notna().sum() > 0:
        priceFigure.add_trace(
            go.Scatter(
                x = metrics["movingAverageFast"].index,
                y = metrics["movingAverageFast"],
                name = "FastMA",
                mode = "lines",
                line = dict(color = "#FFCB05", width = 2)
            )
        )
    if metrics["movingAverageSlow"].notna().sum() > 0:
        priceFigure.add_trace(
            go.Scatter(
                x = metrics["movingAverageSlow"].index,
                y = metrics["movingAverageSlow"],
                name = "SlowMA",
                mode = "lines",
                line = dict(color = "#95a5a6", width = 1.5, dash = "dash")
            )
        )
    priceFigure.update_layout(
        title = inputParameters["symbol"] + " PriceAndTrend " + metrics["trendLabel"],
        xaxis_title = "Date",
        yaxis_title = "Price",
        **umichLayout
    )
    st.plotly_chart(priceFigure, use_container_width = True)

    volatilityFigure = go.Figure()
    volatilityFigure.add_trace(
        go.Scatter(
            x = metrics["rollingVolatilitySeries"].index,
            y = 100 * metrics["rollingVolatilitySeries"],
            name = "Volatility",
            mode = "lines",
            line = dict(color = "#00274C", width = 2)
        )
    )
    if not math.isnan(metrics["lowVolatilityThreshold"]):
        volatilityFigure.add_hline(
            y = 100 * metrics["lowVolatilityThreshold"],
            line = dict(color = "#FFCB05", width = 1.5, dash = "dot"),
            annotation_text = "Low"
        )
    if not math.isnan(metrics["highVolatilityThreshold"]):
        volatilityFigure.add_hline(
            y = 100 * metrics["highVolatilityThreshold"],
            line = dict(color = "#FFCB05", width = 1.5, dash = "dot"),
            annotation_text = "High"
        )
    volatilityFigure.update_layout(
        title = inputParameters["symbol"] + " VolatilityRegime " + metrics["volatilityRegimeLabel"],
        xaxis_title = "Date",
        yaxis_title = "AnnualizedVolatilityPercent",
        **umichLayout
    )
    st.plotly_chart(volatilityFigure, use_container_width = True)

    cumulativeLogReturn = metrics["returnSeries"].cumsum()
    cumulativeReturnSeries = np.exp(cumulativeLogReturn) - 1.0

    cumulativeReturnFigure = go.Figure()
    cumulativeReturnFigure.add_trace(
        go.Scatter(
            x = cumulativeReturnSeries.index,
            y = 100 * cumulativeReturnSeries,
            name = "CumulativeReturn",
            mode = "lines",
            line = dict(color = "#00274C", width = 2)
        )
    )
    cumulativeReturnFigure.update_layout(
        title = inputParameters["symbol"] + " CumulativeReturnCurve",
        xaxis_title = "Date",
        yaxis_title = "CumulativeReturnPercent",
        **umichLayout
    )
    st.plotly_chart(cumulativeReturnFigure, use_container_width = True)

    rollingSharpeFigure = go.Figure()
    if metrics["rollingSharpeShortSeries"].notna().sum() > 0:
        rollingSharpeFigure.add_trace(
            go.Scatter(
                x = metrics["rollingSharpeShortSeries"].index,
                y = metrics["rollingSharpeShortSeries"],
                name = "RollingSharpe63",
                mode = "lines",
                line = dict(color = "#00274C", width = 2)
            )
        )
    if metrics["rollingSharpeLongSeries"].notna().sum() > 0:
        rollingSharpeFigure.add_trace(
            go.Scatter(
                x = metrics["rollingSharpeLongSeries"].index,
                y = metrics["rollingSharpeLongSeries"],
                name = "RollingSharpe126",
                mode = "lines",
                line = dict(color = "#FFCB05", width = 2)
            )
        )
    rollingSharpeFigure.update_layout(
        title = inputParameters["symbol"] + " RollingSharpeTimeline",
        xaxis_title = "Date",
        yaxis_title = "RollingSharpe",
        **umichLayout
    )
    st.plotly_chart(rollingSharpeFigure, use_container_width = True)

    rollingMeanReturnSeries = metrics["rollingMeanReturnSeries"]
    rollingMask = rollingMeanReturnSeries.notna() & metrics["rollingVolatilitySeries"].notna()
    riskReturnFigure = go.Figure()
    if rollingMask.sum() > 0:
        riskReturnFigure.add_trace(
            go.Scatter(
                x = 100 * metrics["rollingVolatilitySeries"][rollingMask],
                y = 100 * rollingMeanReturnSeries[rollingMask],
                mode = "markers",
                name = "RollingRiskReturn",
                marker = dict(size = 6, opacity = 0.6, color = "#00274C")
            )
        )
    riskReturnFigure.update_layout(
        title = inputParameters["symbol"] + " RollingRiskReturnPhaseDiagram",
        xaxis_title = "AnnualizedVolatilityPercent",
        yaxis_title = "AnnualizedMeanReturnPercent",
        **umichLayout
    )
    st.plotly_chart(riskReturnFigure, use_container_width = True)

    rollingDownsideDeviationSeries = metrics["rollingDownsideDeviationSeries"]
    if rollingDownsideDeviationSeries is not None and hasattr(rollingDownsideDeviationSeries, "index") and rollingDownsideDeviationSeries.notna().sum() > 0:
        rollingDownsideFigure = go.Figure()
        rollingDownsideFigure.add_trace(
            go.Scatter(
                x = metrics["rollingVolatilitySeries"].index,
                y = 100 * metrics["rollingVolatilitySeries"],
                name = "RollingVolatility",
                mode = "lines",
                line = dict(color = "#00274C", width = 1.8)
            )
        )
        rollingDownsideFigure.add_trace(
            go.Scatter(
                x = rollingDownsideDeviationSeries.index,
                y = 100 * rollingDownsideDeviationSeries,
                name = "RollingDownsideDeviation",
                mode = "lines",
                line = dict(color = "#FFCB05", width = 1.8)
            )
        )
        rollingDownsideFigure.update_layout(
            title = inputParameters["symbol"] + " VolatilityVsDownsideDeviation",
            xaxis_title = "Date",
            yaxis_title = "AnnualizedPercent",
            **umichLayout
        )
        st.plotly_chart(rollingDownsideFigure, use_container_width = True)

    if metrics["monthlyReturnFrame"] is not None and not metrics["monthlyReturnFrame"].empty:
        monthlyFrame = metrics["monthlyReturnFrame"]
        monthlyFigure = go.Figure()
        monthlyFigure.add_trace(
            go.Bar(
                x = monthlyFrame["YearMonthLabel"],
                y = 100 * monthlyFrame["MonthlyReturn"],
                name = "MonthlyReturnPercent",
                marker = dict(color = "#00274C")
            )
        )
        monthlyFigure.update_layout(
            title = inputParameters["symbol"] + " MonthlyReturnsTimeline",
            xaxis_title = "YearMonth",
            yaxis_title = "MonthlyReturnPercent",
            xaxis = dict(
                tickangle = -45,
                gridcolor = "rgba(225, 228, 232, 0.5)",
                showgrid = True,
                zeroline = False,
                linecolor = "#e1e4e8"
            ),
            yaxis = dict(
                gridcolor = "rgba(225, 228, 232, 0.5)",
                showgrid = True,
                zeroline = False,
                linecolor = "#e1e4e8"
            ),
            margin = dict(t = 50, l = 0, r = 0, b = 80),
            plot_bgcolor = "white",
            paper_bgcolor = "white",
            font = dict(color = "#00274C", family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
            title_font = dict(size = 18, color = "#00274C", family = "sans-serif"),
            showlegend = False,
            hovermode = "x unified"
        )
        st.plotly_chart(monthlyFigure, use_container_width = True)

        heatmapFrame = monthlyFrame.pivot(index = "Year", columns = "Month", values = "MonthlyReturn")
        yearLabels = [str(int(y)) for y in heatmapFrame.index]
        monthLabels = [str(m) for m in heatmapFrame.columns]
        heatmapFigure = go.Figure(
            data = go.Heatmap(
                z = 100 * heatmapFrame.values,
                x = monthLabels,
                y = yearLabels,
                colorscale = "RdBu",
                zmid = 0
            )
        )
        heatmapFigure.update_layout(
            title = inputParameters["symbol"] + " MonthlyReturnHeatmap",
            xaxis_title = "Month",
            yaxis_title = "Year",
            plot_bgcolor = "white",
            paper_bgcolor = "white",
            font = dict(color = "#00274C", family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
            title_font = dict(size = 18, color = "#00274C", family = "sans-serif"),
            margin = dict(t = 50, l = 0, r = 0, b = 0),
            yaxis = dict(type = "category")
        )
        st.plotly_chart(heatmapFigure, use_container_width = True)

    if metrics["volatilityRegimeSeries"] is not None:
        regimeSeries = metrics["volatilityRegimeSeries"].dropna()
        if not regimeSeries.empty:
            regimeFigure = go.Figure(
                data = go.Heatmap(
                    z = [regimeSeries.values],
                    x = regimeSeries.index,
                    y = ["VolatilityRegime"],
                    colorscale = [
                        [0.0, "#2ecc71"],
                        [0.5, "#f1c40f"],
                        [1.0, "#e74c3c"]
                    ],
                    colorbar = dict(
                        ticks = "outside",
                        tickvals = [0, 1, 2],
                        ticktext = ["Low", "Medium", "High"],
                        title = "Regime"
                    )
                )
            )
            regimeFigure.update_layout(
                title = inputParameters["symbol"] + " VolatilityRegimeTimeline",
                xaxis_title = "Date",
                yaxis_title = "",
                plot_bgcolor = "white",
                paper_bgcolor = "white",
                font = dict(color = "#00274C", family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
                title_font = dict(size = 18, color = "#00274C", family = "sans-serif"),
                margin = dict(t = 50, l = 0, r = 0, b = 0)
            )
            st.plotly_chart(regimeFigure, use_container_width = True)

    st.subheader("Tail Risk And Regimes")

    returnSeries = metrics["returnSeries"]
    varThreshold = metrics["varValue"]
    cvarThreshold = metrics["cvarValue"]

    returnsVaRFigure = go.Figure()
    returnsVaRFigure.add_trace(
        go.Scatter(
            x = returnSeries.index,
            y = 100 * returnSeries,
            name = "DailyLogReturnPercent",
            mode = "lines",
            line = dict(color = "#00274C", width = 1.5)
        )
    )
    returnsVaRFigure.add_hline(
        y = 100 * varThreshold,
        line = dict(color = "#e67e22", width = 2, dash = "dash"),
        annotation_text = "VaR"
    )
    returnsVaRFigure.add_hline(
        y = 100 * cvarThreshold,
        line = dict(color = "#c0392b", width = 2, dash = "dot"),
        annotation_text = "CVaR"
    )
    returnsVaRFigure.update_layout(
        title = inputParameters["symbol"] + " ReturnsWithVaRAndCVaR",
        xaxis_title = "Date",
        yaxis_title = "DailyReturnPercent",
        **umichLayout
    )
    st.plotly_chart(returnsVaRFigure, use_container_width = True)

    varBreachRollingCountSeries = metrics["varBreachRollingCountSeries"]
    if varBreachRollingCountSeries is not None and varBreachRollingCountSeries.notna().sum() > 0:
        varBreachFigure = go.Figure()
        varBreachFigure.add_trace(
            go.Scatter(
                x = varBreachRollingCountSeries.index,
                y = varBreachRollingCountSeries,
                name = "RollingVaRBreaches",
                mode = "lines",
                line = dict(color = "#c0392b", width = 2)
            )
        )
        varBreachFigure.update_layout(
            title = inputParameters["symbol"] + " RollingVaRBreachCounts",
            xaxis_title = "Date",
            yaxis_title = "BreachesInWindow",
            **umichLayout
        )
        st.plotly_chart(varBreachFigure, use_container_width = True)

    rollingReturnQuantileLowSeries = metrics["rollingReturnQuantileLowSeries"]
    rollingReturnQuantileHighSeries = metrics["rollingReturnQuantileHighSeries"]
    if rollingReturnQuantileLowSeries is not None and rollingReturnQuantileLowSeries.notna().sum() > 0:
        quantileFigure = go.Figure()
        quantileFigure.add_trace(
            go.Scatter(
                x = rollingReturnQuantileLowSeries.index,
                y = 100 * rollingReturnQuantileLowSeries,
                name = "Rolling5thPercentile",
                mode = "lines",
                line = dict(color = "#e74c3c", width = 1.8)
            )
        )
        if rollingReturnQuantileHighSeries is not None:
            quantileFigure.add_trace(
                go.Scatter(
                    x = rollingReturnQuantileHighSeries.index,
                    y = 100 * rollingReturnQuantileHighSeries,
                    name = "Rolling95thPercentile",
                    mode = "lines",
                    line = dict(color = "#27ae60", width = 1.8)
                )
            )
        quantileFigure.update_layout(
            title = inputParameters["symbol"] + " RollingReturnQuantileBand",
            xaxis_title = "Date",
            yaxis_title = "ReturnPercent",
            **umichLayout
        )
        st.plotly_chart(quantileFigure, use_container_width = True)

    rollingSortinoSeries = metrics["rollingSortinoSeries"]
    if rollingSortinoSeries is not None and rollingSortinoSeries.notna().sum() > 0:
        jointMask = (
            metrics["rollingSharpeShortSeries"].notna()
            & rollingSortinoSeries.notna()
        )
        sortinoSharpeFigure = go.Figure()
        if jointMask.sum() > 0:
            sortinoSharpeFigure.add_trace(
                go.Scatter(
                    x = metrics["rollingSharpeShortSeries"][jointMask],
                    y = rollingSortinoSeries[jointMask],
                    mode = "markers",
                    name = "RollingSharpeVsSortino",
                    marker = dict(size = 6, opacity = 0.6, color = "#00274C")
                )
            )
        sortinoSharpeFigure.update_layout(
            title = inputParameters["symbol"] + " RollingSharpeVsSortino",
            xaxis_title = "RollingSharpe63",
            yaxis_title = "RollingSortino",
            **umichLayout
        )
        st.plotly_chart(sortinoSharpeFigure, use_container_width = True)

    if metrics["returnRegimeSeries"] is not None:
        rrSeries = metrics["returnRegimeSeries"].dropna()
        if not rrSeries.empty:
            returnRegimeFigure = go.Figure(
                data = go.Heatmap(
                    z = [rrSeries.values],
                    x = rrSeries.index,
                    y = ["ReturnRegime"],
                    colorscale = [
                        [0.0, "#e74c3c"],
                        [0.5, "#bdc3c7"],
                        [1.0, "#2ecc71"]
                    ],
                    colorbar = dict(
                        ticks = "outside",
                        tickvals = [-1, 0, 1],
                        ticktext = ["Down", "Flat", "Up"],
                        title = "Regime"
                    )
                )
            )
            returnRegimeFigure.update_layout(
                title = inputParameters["symbol"] + " ReturnRegimeTimeline",
                xaxis_title = "Date",
                yaxis_title = "",
                plot_bgcolor = "white",
                paper_bgcolor = "white",
                font = dict(color = "#00274C", family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
                title_font = dict(size = 18, color = "#00274C", family = "sans-serif"),
                margin = dict(t = 50, l = 0, r = 0, b = 0)
            )
            st.plotly_chart(returnRegimeFigure, use_container_width = True)

    if metrics["drawdownEventsFrame"] is not None and not metrics["drawdownEventsFrame"].empty:
        ddFrame = metrics["drawdownEventsFrame"].copy()
        ddFrame["DepthPercent"] = 100 * ddFrame["Depth"]
        drawdownScatterFigure = go.Figure()
        drawdownScatterFigure.add_trace(
            go.Scatter(
                x = ddFrame["Duration"],
                y = ddFrame["DepthPercent"],
                mode = "markers",
                name = "DrawdownEvents",
                marker = dict(size = 8, opacity = 0.7, color = "#c0392b")
            )
        )
        drawdownScatterFigure.update_layout(
            title = inputParameters["symbol"] + " DrawdownDepthVsDuration",
            xaxis_title = "DurationDays",
            yaxis_title = "DepthPercent",
            **umichLayout
        )
        st.plotly_chart(drawdownScatterFigure, use_container_width = True)

    st.subheader("Risk & Tail Analysis")

    tailConfidence = int((1 - inputParameters["alpha"]) * 100)
    riskCol1, riskCol2, riskCol3, riskCol4 = st.columns(4)
    riskCol1.metric(str(tailConfidence) + "% VaR", "{:.2f}%".format(100 * metrics["varValue"]))
    riskCol2.metric(str(tailConfidence) + "% CVaR", "{:.2f}%".format(100 * metrics["cvarValue"]))
    riskCol3.metric("DownsideDeviation", "{:.2f}%".format(100 * metrics["downsideDeviation"]))
    riskCol4.metric("UlcerIndex", "{:.2f}%".format(100 * metrics["ulcerIndex"]))

    drawCol1, drawCol2, drawCol3, drawCol4 = st.columns(4)
    drawCol1.metric("MaxDrawdown", "{:.1f}%".format(100 * metrics["maximumDrawdown"]))
    drawCol2.metric("CurrentDrawdown", "{:.1f}%".format(100 * metrics["currentDrawdown"]))
    drawCol3.metric("LongestDrawdown", str(metrics["drawdownDays"]) + " days")
    drawCol4.metric("AvgRecovery", "{:.0f} days".format(metrics["avgRecoveryTime"]))

    distCol1, distCol2, distCol3, distCol4 = st.columns(4)
    distCol1.metric("Skewness", "{:.2f}".format(stats.skew(metrics["returnSeries"])))
    distCol2.metric("ExcessKurtosis", "{:.2f}".format(stats.kurtosis(metrics["returnSeries"], fisher = True)))
    distCol3.metric("GainLossRatio", "{:.2f}".format(metrics["gainLossRatio"]))
    jarqueStatistic, jarquePValue = stats.jarque_bera(metrics["returnSeries"])
    distCol4.metric("JB p-value", "{:.4f}".format(jarquePValue))

    drawdownFigure = go.Figure()
    drawdownFigure.add_trace(
        go.Scatter(
            x = metrics["drawdownSeries"].index,
            y = 100 * metrics["drawdownSeries"],
            fill = "tozeroy",
            name = "Drawdown",
            mode = "lines",
            line = dict(color = "#00274C", width = 1.5),
            fillcolor = "rgba(0, 39, 76, 0.1)"
        )
    )
    drawdownTitle = inputParameters["symbol"] + " DrawdownAnalysis"
    drawdownFigure.update_layout(
        title = drawdownTitle,
        xaxis_title = "Date",
        yaxis_title = "DrawdownPercent",
        **umichLayout
    )
    st.plotly_chart(drawdownFigure, use_container_width = True)

    distributionFigure = go.Figure()
    distributionFigure.add_trace(
        go.Bar(
            x = metrics["histogramMidpoints"],
            y = metrics["histogramHeights"],
            name = "Returns",
            marker = dict(
                color = "rgba(0, 39, 76, 0.6)",
                line = dict(color = "#00274C", width = 1)
            )
        )
    )
    distributionFigure.add_trace(
        go.Scatter(
            x = metrics["evaluationGrid"],
            y = metrics["pdfValues"],
            name = "StudentT",
            mode = "lines",
            line = dict(color = "#FFCB05", width = 3)
        )
    )
    distributionFigure.update_layout(
        title = inputParameters["symbol"] + " ReturnDistribution",
        xaxis_title = "DailyLogReturn",
        yaxis_title = "Density",
        **umichLayout
    )
    st.plotly_chart(distributionFigure, use_container_width = True)

    st.subheader("Monte Carlo Simulation")

    simCol1, simCol2, simCol3, simCol4 = st.columns(4)
    simCol1.metric("Simulations", "{:,}".format(inputParameters["histogramPaths"]))
    simCol2.metric("TimeHorizon", str(inputParameters["simulationDays"]) + " days")
    simCol3.metric("MedianPrice", "$" + "{:.2f}".format(np.median(metrics["simulationFinalsArray"])))
    simCol4.metric(
        "95% CI Width",
        "{:.1f}%".format(
            100
            * (
                np.percentile(metrics["simulationFinalsArray"], 97.5)
                - np.percentile(metrics["simulationFinalsArray"], 2.5)
            )
            / priceSeries.iloc[-1]
        )
    )

    simLayout = dict(
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        font = dict(color = "#00274C", family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        title_font = dict(size = 18, color = "#00274C", family = "sans-serif"),
        showlegend = False,
        hovermode = "x unified",
        xaxis = dict(
            gridcolor = "rgba(225, 228, 232, 0.5)",
            showgrid = True,
            zeroline = False,
            linecolor = "#e1e4e8"
        ),
        yaxis = dict(
            gridcolor = "rgba(225, 228, 232, 0.5)",
            showgrid = True,
            zeroline = False,
            linecolor = "#e1e4e8"
        ),
        margin = dict(t = 50, l = 0, r = 0, b = 0)
    )

    pathFigure = go.Figure()
    visiblePathCount = metrics["simulationPathsVisible"].shape[1]
    for pathIndex in range(visiblePathCount):
        colorIntensity = 0.2 + (pathIndex / visiblePathCount) * 0.6
        lineColor = f"rgba(0, 39, 76, {colorIntensity})"
        pathFigure.add_trace(
            go.Scatter(
                x = np.arange(1, inputParameters["simulationDays"] + 1),
                y = metrics["simulationPathsVisible"][:, pathIndex],
                mode = "lines",
                line = dict(color = lineColor, width = 1),
                showlegend = False
            )
        )
    pathFigure.update_layout(
        title = inputParameters["symbol"] + " SimulatedPricePaths",
        xaxis_title = "Day",
        yaxis_title = "SimulatedPrice",
        **simLayout
    )
    st.plotly_chart(pathFigure, use_container_width = True)

    histogramFigure = go.Figure()
    histogramFigure.add_trace(
        go.Histogram(
            x = metrics["trimmedFinals"],
            nbinsx = 30,
            marker = dict(
                color = "#00274C",
                line = dict(color = "#FFCB05", width = 1)
            )
        )
    )
    histogramTitle = inputParameters["symbol"] + " TerminalPriceDistribution"
    histogramFigure.update_layout(
        title = histogramTitle,
        xaxis_title = "SimulatedTerminalPrice",
        yaxis_title = "Frequency",
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        font = dict(color = "#00274C", family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        title_font = dict(size = 18, color = "#00274C", family = "sans-serif"),
        showlegend = False,
        hovermode = "x unified",
        margin = dict(t = 50, l = 0, r = 0, b = 0)
    )
    st.plotly_chart(histogramFigure, use_container_width = True)

    st.markdown(
        """
        <div style='text-align: center; margin-top: 40px; padding: 30px; color: #6c757d; font-size: 0.9em; border-top: 1px solid #e1e4e8;'>
            <span style='display: inline-block; color: #00274C; font-weight: 700; border-bottom: 2px solid #FFCB05; padding-bottom: 2px;'>GO BLUE</span> 
            | by Avery Cloutier, Zara Masood, & Jeffrey Prachick
        </div>
        """,
        unsafe_allow_html = True
    )


inputParameters = readSidebarInputs()
priceSeries = loadPriceSeries(inputParameters)
if priceSeries is not None and len(priceSeries) > 0:
    metrics = computeAllMetrics(priceSeries, inputParameters)
    renderDashboard(priceSeries, inputParameters, metrics)
else:
    st.info('No price series loaded. Adjust the sidebar and try again.')
