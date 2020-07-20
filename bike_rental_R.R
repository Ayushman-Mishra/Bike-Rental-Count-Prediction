#Clean the environment
rm(list = ls())

#Set working directory
setwd("C:\\Users\\M1053735\\Pictures\\bikeRental")
getwd()


# loading Libraries
x = c("ggplot2", "corrgram", "usdm", "caret", "DMwR", "rpart", "randomForest",'xgboost','moments','car','DataCombine','rsq')


# install.packages if not
#lapply(x, install.packages)

# load Packages
lapply(x, require, character.only = TRUE)
rm(x)

# loading dataset
df = read.csv('day.csv',header = TRUE,na.strings = c('',' ','NA'),row.names="dteday")[, -c(1)]
#dropping instatnt because it is unstatiscally 


# Structure of data
str(df)

# Summary of data
summary(df)

# Viewing the data
head(df,5)

######################## EDA, Missing value and Outlier analysis  #########


# converting variables to their proper datatypes
categorical_variables = c("season","yr","mnth","holiday","weekday","workingday","weathersit")

for(i in categorical_variables){
  df[,i] = as.factor(df[,i])
}

#removing instant(statistically not useful)
df$instant <- NULL

## check for missing values
apply(df,2, function(x){ sum(is.na(x))})
#No Missing Values are found


#Dropping casual and registered column because they are not the target variable and the sum of both is the target variable i.e. cnt
df[,c("casual","registered")] <- list(NULL)

continuous_variables = c('temp', 'atemp', 'hum', 'windspeed','cnt')

#function to vizualize the continuity of continuous variables
continuos_Vars_display <- function(cntnus){
  ggplot(df) +
    geom_histogram(aes(x = cntnus, y = ..density..),fill='green',colour='black') +
    geom_density(aes(x = cntnus, y = ..density..)) 
    
}
continuos_Vars_display(df$temp)
continuos_Vars_display(df$atemp)
continuos_Vars_display(df$hum)
continuos_Vars_display(df$windspeed)
continuos_Vars_display(df$cnt)

#check skewness of the target variable
print(skewness(df$cnt))  ## => -0.04725556
# The rule of thumb seems to be:
# If the skewness is between -0.5 and 0.5, the data are fairly symmetrical. so we can say that our cnt is normally distributed

##Plotting Box Plots
for(i in 1:length(continuous_variables)){
  assign(paste0("gn",i), ggplot(data = df, aes_string(x = "cnt", y = continuous_variables[i]))+
           stat_boxplot(geom = "errorbar",width = 0.5)+
           geom_boxplot(outlier.colour = 'red',fill='grey',outlier.shape = 18,outlier.size = 4)+
           labs(y=continuous_variables[i],x='cnt')+
           ggtitle(paste("Box plot of",continuous_variables[i])))
  
}

gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)

#as we can see hum and windspeed has outliers and now we will remove them
#make a copy of original data
df_copy = df

outliers_vars = c('hum','windspeed')

for(i in outliers_vars){
  temp = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  df[,i][df[,i] %in% temp] = NA
}

# check for missing values
apply(df,2, function(x){ sum(is.na(x))})
#we found 2 and 13 outliers in hum and windspeed respectively

#outliers removal
df = knnImputation(data = df, k = 3)

# create backup of data
df_no_outlies = df


#check relationship b/w categorical and target variable
for(i in 1:length(categorical_variables)){
  assign(paste0("s",i),ggplot(data = df, aes_string(fill = "cnt", x = categorical_variables[i])) +
    geom_bar(position = "dodge") + ggtitle(paste("cnt vs ",categorical_variables[i])))
}
gridExtra::grid.arrange(s1,s2,s3,ncol=3)
gridExtra::grid.arrange(s4,s5,s6,ncol=3)
gridExtra::grid.arrange(s7,ncol=1)

###################### Feature Extraction ################################
# correlation plot
corrgram(df[,continuous_variables],order = FALSE,
         upper.panel=panel.pie, text.panel=panel.txt, main='Correlation Plot')
# as we can see temp and atemp are highly positively correlated with each other


## Chi-squared Test of Independence
for(i in categorical_variables){
  for(j in categorical_variables){
    print(i)
    print(j)
    print(chisq.test(table(df[,i], df[,j]))$p.value)
  }
}
# check VIF
vif(df[,8:11])
vifcor(df[,8:11])

# ANOVA test 

for (i in categorical_variables) {
  print(i)
  print(summary(aov(df$cnt ~df[,i], df)))
}

#Taking correlation plot and vif Results into cosideration, Removing variables atemp beacuse it is highly correlated with temp,
#Taking Anova Test Results into consideration, Removing weekday, holiday because they don't contribute much to the independent cariable
## Dimension Reduction
df=subset(df,select=-c(atemp,holiday,weekday))



#Normality check for target variable
qqnorm(df$cnt); qqline(df$cnt)



# Normalization of cnt
df$cnt = (df$cnt - min(df$cnt)) / (max(df$cnt) - min(df$cnt))
#no changes has been seen in qqnorm before and after normalisation of cnt variable

#create dummies for categorical variables
dmmy_frmula = dummyVars(~., df)
df = data.frame(predict(dmmy_frmula, df))

############ Splitting df into train and test ###################
set.seed(12345)
train_index = createDataPartition(df$cnt, p = 0.8, list=FALSE)
train = df[train_index,]
test = df[-train_index,]

rmExcept(c("test","train",'df','df_no_outlies'))

################ Model Development ######################
# Linear Regression
linear_regressor = lm(cnt ~.,data = train)
summary(linear_regressor)

pred = predict(linear_regressor,test[,-27])

regr.eval(test[,27],preds = pred)
#      mae          mse         rmse         mape 
# 0.069017507   0.009433239  0.097124863  0.206684169 

#calculate R-Squared value
rsq(fitObj = linear_regressor,adj = TRUE,data = train)
# 0.8418282

############## Decision Tree Model ###############
tree = rpart(cnt ~ ., data=train, method = "anova")
summary(tree)

pred_dt = predict(tree, test[,-27])

regr.eval(test[,27],preds = pred_dt)
#      mae         mse         rmse         mape 
# 0.07537033   0.01037191   0.10184259    0.21901332 

# calculate R-Square value
rss_dt = sum((pred_dt - test$cnt) ^ 2)
tss_dt = sum((test$cnt - mean(test$cnt)) ^ 2)
rsq_dt = 1 - rss_dt/tss_dt
# 0.7605108

#############  Random forest Model #####################
rf_model = randomForest(cnt ~.,data=train, importance = TRUE, ntree=500)
summary(rf_model)

pred_rm = predict(rf_model,test[-27])

regr.eval(test[,27],preds = pred_rm)
#    mae            mse        rmse        mape 
# 0.056957955  0.006789371  0.082397642 0.161405671 

# calculate R-Square value
rss_rf = sum((pred_rm - test$cnt) ^ 2)
tss_rf = sum((test$cnt - mean(test$cnt)) ^ 2)
rsq_rf = 1 - rss_rf/tss_rf
# 0.8432323


############  XGBOOST Model ###########################
train_data_matrix = as.matrix(sapply(train[-27],as.numeric))
test_data_matrix = as.matrix(sapply(test[-27],as.numeric))

xgb = xgboost(data = train_data_matrix,label = train$cnt, nrounds = 13,verbose = TRUE)

pred_xgb = predict(xgb,test_data_matrix)

regr.eval(test[,27],preds = pred_xgb)
#  mae           mse          rmse       mape 
# 0.060146910 0.006860217 0.082826426 0.164038736 


#calculate R-Square value
rss_xgb = sum((pred_xgb - test$cnt) ^ 2)
tss_xgb = sum((test$cnt - mean(test$cnt)) ^ 2)
rsq_xgb = 1 - rss_xgb/tss_xgb
#0.8415965


#########################################################################################################################################
# from the above models we can conclude that XGBOOST Model is the best fit for this problem.