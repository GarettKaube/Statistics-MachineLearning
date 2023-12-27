library(tidyverse)
library(fredr)
library(fpp2)
library(dplyr)
library(mFilter)
library(pls)
library(forecast)
library(lmtest)
library(sandwich)
library(hash)


fredr_set_key("9a048a276f1a939a1e64c77f214e5684")

drop_cols = function(df, name) {
  #""" Drops series_id, realtime_start, realtime_end columns
  #"""
  
  df = df %>%  select(-c(series_id, realtime_start, realtime_end)) %>%
    rename(!!sym(name) := value)
  return(df)
}


get_data = function(hash_funct, start_=as.Date("2018-01-01")) {
  series_ids = keys(hash_funct)
  # Download time series 
  series_list = list()
  for (i in seq_along(series_ids)) {
    series = fredr(series_id=series_ids[i], observation_start=start_, frequency='m') %>% 
      drop_cols(name=hash_funct[[series_ids[i]]])
  
    series_list = append(series_list, list(series))
  }
  
  # merge the time series
  merged = merge(series_list[1], series_list[2], by="date")
  for (i in 3:length(series_list)) {
    merged = merge(merged, series_list[i], by="date")
  }
  
  return(merged)
  
}



extract_year = function(x) {
  return(format(as.Date(x), format = "%Y"))
}

data_pipeline = function() {
  # Federal Funds Effective Rate, Bank of Canada Overnight Rate, Coinbase Bitcoin
  # Hash function format is h[["FRED_ID"]] = "User choice name"
  h = hash()
  h[["FEDFUNDS"]] = "FEDFUNDS"
  h[["IRSTCB01CAM156N"]] = "OVERNIGHT"
  h[["CBBTCUSD"]] = "BITCOIN"
  h[["DEXCAUS"]] = "CADUSD"
  h[["EXUSEU"]] = "EUROUSD"
  h[["SP500"]] = "SP500"
  
  data = get_data(h)
  
  # Difference 
  for (col in list("FEDFUNDS", "BITCOIN", "OVERNIGHT")) {
    name = paste0(col, "_DIFF")
    data = mutate(data, {{name}} := c(NA, data[, col] %>% log() %>% diff()))
  }
  
  # Create year variable
  data[, "year"] = sapply(data[,"date"], FUN = extract_year)
  
  # Lag FEDFUNDS
  n_lags = 6
  for (i in 1:6){
    name = paste0("FEDFUNDS_LAGGED",i)
    data = mutate(data, !!sym(name) := data[, "FEDFUNDS"] %>% lag(n=i))
  }
  
  # Remove NA
  data = na.omit(data)
  return(data)
  
}

data = data_pipeline()
head(data)


data.ts = ts(select(data, -c("date")), start = 2018, freq=12)

standardize_var = function(data, var, new_var_name=NULL) {
  # data: data.frame.
  # var: string.
  # new_var_name: If user does not want to replace original variable with the standardized one.
  var_data = data[, var]
  
  if (is.null(new_var_name)){
    data[, var] = (var_data - mean(var_data)) / sd(var_data)
  }
  else {
    data[, new_var_name] = (var_data - mean(var_data)) / sd(var_data)
  }
  
  return(data)
}

# Scale Bitcoin price to plot bitcoin price and FEDFUNDS together
data = standardize_var(data, "BITCOIN", "SCALED_BITCOIN")
ggplot(data=data) +
  geom_line(aes(x=date, y = FEDFUNDS)) +
  geom_line(aes(x=date, y = SCALED_BITCOIN))


# Scale SP500 price to plot bitcoin price and FEDFUNDS together
data = standardize_var(data, "SP500", "SP500_SCALED")
ggplot(data=data) +
  geom_line(aes(x=date, y = FEDFUNDS)) +
  geom_line(aes(x=date, y = SP500_SCALED))


cor(data.ts)


# Linear correlation between S&P500 and FED Rate
cor(data.ts)["FEDFUNDS", "SP500"]
cor.test(data.ts[, "FEDFUNDS"], data.ts[, "SP500"], method = "pearson")

# Linear correlation between Bitcoin price and FED Rate
cor(data.ts)["FEDFUNDS", "BITCOIN"]
cor.test(data.ts[, "FEDFUNDS"], data.ts[, "BITCOIN"], method = "pearson")

# Plot Overnight rate
ggplot(data=data, aes(x=date, y=OVERNIGHT)) +
  geom_line()


# Scatter plot of FEDFUNDS vs BITCOIN price
# Clearly there is a non-linear relationship
ggplot(data = data, aes(x=FEDFUNDS, y=BITCOIN_DIFF)) +
  geom_point(aes(color=year, size=3)) +
  geom_smooth() +
  theme_minimal()

ggplot(data = data, aes(x=CADUSD, y=BITCOIN)) +
  geom_point(aes(color=year, size=3)) +
  geom_smooth() +
  theme_minimal()

ggplot(data = data, aes(x=FEDFUNDS, y=SP500)) +
  geom_point(aes(color=year, size=3)) +
  geom_smooth() +
  theme_minimal()


# Check multicolinearity
predictors = c("FEDFUNDS", "CADUSD", "EUROUSD")
get_VIF = function(predictors) {
  # Initialize VIF data frame
  VIF = data.frame()
  for (i in 1:(length(predictors))) {
    target = predictors[i]
    
    # Get predictors
    predictors_modified = predictors[-c(i)]
    
    # Create regression formulas
    predictor_str = predictors_modified[1]
    predictors_modified = predictors_modified[-c(1)]
    for (pred in predictors_modified) {
      predictor_str = predictor_str %>% paste("+") %>% paste(pred)
    }
  
    formula = paste(target, "~") %>% paste(predictor_str) 
    print(paste("Model: ",formula))
    
    # Regress, get R-Squared and calculate Variance factors
    r2 = summary(lm({{formula}}, data = data))$r.squared
    VIF[1, paste(formula, "|")] =  1/(1-r2)
    
  }
  return(VIF)
}
get_VIF(predictors)


# Second order polynomial regression
formula = "BITCOIN ~ FEDFUNDS + I(FEDFUNDS**2) + CADUSD + EUROUSD"
model = lm({{formula}}, data=data)
summary(model)


# Diagnostics
# The residuals look to be centered around 0 but we clearly have heteroskedasticity for FEDFUNDS VS RESIDUALS
# and possibly EUROUSD VS RESIDUALS
# The model is valid as long as we account for the heteroskedasticity.
residual_plot = function(var){
  df = data.frame(var = data[, var] ,res=residuals(model))
  ggplot(data = df, aes(x=var, y=res)) +
    geom_point(color='skyblue') +
    theme_minimal() +
    labs(y= "Residuals", x = var)
    
  
}
residual_plot("FEDFUNDS")
residual_plot("CADUSD")
residual_plot("EUROUSD")



# Account for heteroskedasticity
# Robust t test
coeftest(model, vcov = vcovHC(model, type = "HC0"))


# Testing regression anatomy http://fmwww.bc.edu/repec/bocode/r/reganat.pdf

# Regress FEDFUNDS on all other predictors and get residuals
x1 = lm(FEDFUNDS ~ I(FEDFUNDS**2) + CADUSD + EUROUSD, data=data)
resx1 = residuals(x1)

# Calculate covariance between Bitcoin price and residuals
joint_data = data.frame(BITCOIN = data[,"BITCOIN"], resx1 = resx1)
covariance = cov(joint_data)["BITCOIN", "resx1"]

# Calculate the parameter
b1 = covariance / var(resx1)
b1






