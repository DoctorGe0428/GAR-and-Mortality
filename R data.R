###决策树

# install.packages("tidymodels")
library(tidymodels)
source("tidyfuncs4cls2_v18.R")

# 多核并行
library(doParallel)
registerDoParallel(
  makePSOCKcluster(
    max(1, (parallel::detectCores(logical = F))-1)
  )
)

# 读取数据
# file.choose()
Heart <- 机器学习数据
colnames(Heart) 
# 修正变量类型
# 将分类变量转换为factor
for(i in c(10,11,23,24,25)){ 
  Heart[[i]] <- factor(Heart[[i]])
}

# 删除无关变量在此处进行
Heart$Id <- NULL
# 删除含有缺失值的样本在此处进行，填充缺失值在后面
Heart <- na.omit(Heart)
# Heart <- Heart %>%
#   drop_na(Thal)
str(Heart)
# 数据概况
skimr::skim(Heart)    

# 设定阳性类别和阴性类别
yourpositivelevel <- "Yes"
yournegativelevel <- "No"
# 转换因变量的因子水平，将阳性类别设定为第二个水平
levels(Heart$AHD)
table(Heart$AHD)
Heart$AHD <- factor(
  Heart$AHD,
  levels = c(yournegativelevel, yourpositivelevel)
)
levels(Heart$AHD)
table(Heart$AHD)

##############################################################

# 数据拆分
set.seed(42)
datasplit <- initial_split(Heart, prop = 0.75, strata = AHD)
traindata <- training(datasplit) %>%
  sample_n(nrow(.))
testdata <- testing(datasplit) %>%
  sample_n(nrow(.))

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata, v = 5, strata = AHD)
folds

# 数据预处理配方
datarecipe_dt <- recipe(formula = AHD ~ ., traindata)
datarecipe_dt


# 设定模型
model_dt <- decision_tree(
  mode = "classification",
  engine = "rpart",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_args(model=T)
model_dt

# workflow
wk_dt <- 
  workflow() %>%
  add_recipe(datarecipe_dt) %>%
  add_model(model_dt)
wk_dt

##############################################################
# 贝叶斯优化超参数
set.seed(42)
tune_dt <- wk_dt %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_dt <- tune_dt %>%
  collect_metrics()
eval_tune_dt

# 图示
# autoplot(tune_dt)
eval_tune_dt %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost_complexity', values = ~cost_complexity),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "DT HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_dt <- tune_dt %>%
  select_by_one_std_err(metric = "roc_auc", desc(cost_complexity))
hpbest_dt

# 采用最优超参数组合训练最终模型
set.seed(42)
final_dt <- wk_dt %>%
  finalize_workflow(hpbest_dt) %>%
  fit(traindata)
final_dt

##################################################################

# 训练集预测评估
predtrain_dt <- eval4cls2(
  model = final_dt, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "DT", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# pROC包auc值及其置信区间
pROC::auc(predtrain_dt$proc)
pROC::ci.auc(predtrain_dt$proc)

# 预测评估测试集预测评估
predtest_dt <- eval4cls2(
  model = final_dt, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "DT", 
  datasetname = "testdata",
  cutoff = predtrain_dt$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_dt$proc)
pROC::ci.auc(predtest_dt$proc)

# ROC比较检验
pROC::roc.test(predtrain_dt$proc, predtest_dt$proc)


# 合并训练集和测试集上ROC曲线
predtrain_dt$rocresult %>%
  bind_rows(predtest_dt$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_dt$prresult %>%
  bind_rows(predtest_dt$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_dt$caliresult %>%
  bind_rows(predtest_dt$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_dt$metrics %>%
  bind_rows(predtest_dt$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_dt <- bestcv4cls2(
  wkflow = wk_dt,
  tuneresult = tune_dt,
  hpbest = hpbest_dt,
  yname = "AHD",
  modelname = "DT",
  v = 5,
  positivelevel = yourpositivelevel
)




# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_dt2 <- final_dt %>%
  extract_fit_engine()
final_dt2


######################## DALEX解释对象

explainer_dt <- DALEXtra::explain_tidymodels(
  final_dt, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "DT"
)
# 变量重要性
vip_dt <- viplot(explainer_dt)
vipdata_dt <- vip_dt$data
vip_dt$plot


# 保存评估结果
save(datarecipe_dt,
     model_dt,
     wk_dt,
     #hpgrid_dt, # 如果采用贝叶斯优化则删掉这一行
     tune_dt,
     predtrain_dt,
     predtest_dt,
     evalcv_dt,
     vipdata_dt,
     file = ".\\cls2\\evalresult_dt.RData")

###随机森林
# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---随机森林
# https://parsnip.tidymodels.org/reference/details_rand_forest_randomForest.html

##############################################################


# 数据预处理配方
datarecipe_rf <- recipe(formula = AHD ~ ., traindata)
datarecipe_rf


# 设定模型
model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest", # ranger
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_args(importance = T)
model_rf

# workflow
wk_rf <- 
  workflow() %>%
  add_recipe(datarecipe_rf) %>%
  add_model(model_rf)
wk_rf

##############################################################
#########################  超参数寻优2选1-贝叶斯优化

# 更新超参数范围
param_rf <- model_rf %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)),
         trees = trees(c(100, 1000)),
         min_n = min_n(c(7, 55)))

# 贝叶斯优化超参数
set.seed(42)
tune_rf <- wk_rf %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_rf,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_rf <- tune_rf %>%
  collect_metrics()
eval_tune_rf

# 图示
# autoplot(tune_rf)
eval_tune_rf %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n)
    )
  ) %>%
  plotly::layout(title = "RF HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_rf <- tune_rf %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_rf

# 采用最优超参数组合训练最终模型
set.seed(42)
final_rf <- wk_rf %>%
  finalize_workflow(hpbest_rf) %>%
  fit(traindata)
final_rf

##################################################################

# 训练集预测评估
predtrain_rf <- eval4cls2(
  model = final_rf, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "RF", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# pROC包auc值及其置信区间
pROC::auc(predtrain_rf$proc)
pROC::ci.auc(predtrain_rf$proc)

# 预测评估测试集预测评估
predtest_rf <- eval4cls2(
  model = final_rf, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "RF", 
  datasetname = "testdata",
  cutoff = predtrain_rf$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_rf$proc)
pROC::ci.auc(predtest_rf$proc)

# ROC比较检验
pROC::roc.test(predtrain_rf$proc, predtest_rf$proc)


# 合并训练集和测试集上ROC曲线
predtrain_rf$rocresult %>%
  bind_rows(predtest_rf$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_rf$prresult %>%
  bind_rows(predtest_rf$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_rf$caliresult %>%
  bind_rows(predtest_rf$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_rf$metrics %>%
  bind_rows(predtest_rf$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_rf <- bestcv4cls2(
  wkflow = wk_rf,
  tuneresult = tune_rf,
  hpbest = hpbest_rf,
  yname = "AHD",
  modelname = "RF",
  v = 5,
  positivelevel = yourpositivelevel
)




# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_rf2 <- final_rf %>%
  extract_fit_engine()
final_rf2



# 变量重要性
randomForest::importance(final_rf2)
randomForest::varImpPlot(
  final_rf2, 
  main = "变量重要性", 
  family = "serif"
)


######################## DALEX解释对象

explainer_rf <- DALEXtra::explain_tidymodels(
  final_rf, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "RF"
)
# 变量重要性
vip_rf <- viplot(explainer_rf)
vipdata_rf <- vip_rf$data
vip_rf$plot
######################################################################

# 保存评估结果
save(datarecipe_rf,
     model_rf,
     wk_rf,
     #hpgrid_rf,   # 如果采用贝叶斯优化则替换为 param_rf
     tune_rf,
     predtrain_rf,
     predtest_rf,
     evalcv_rf,
     vipdata_rf,
     file = ".\\cls2\\evalresult_rf.RData")

###xgboost

##############################################################



# 数据预处理配方
datarecipe_xgboost <- recipe(formula = AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors())
datarecipe_xgboost


# 设定模型
model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) %>%
  set_args(validation = 0.2,
           event_level = "second")
model_xgboost

# workflow
wk_xgboost <- 
  workflow() %>%
  add_recipe(datarecipe_xgboost) %>%
  add_model(model_xgboost)
wk_xgboost

##############################################################


###超参数寻优2选1-贝叶斯优化

# 更新超参数范围
param_xgboost <- model_xgboost %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)))

# 贝叶斯优化超参数
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_xgboost,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost

# 图示
# autoplot(tune_xgboost)
eval_tune_xgboost %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction),
      list(label = 'sample_size', values = ~sample_size)
    )
  ) %>%
  plotly::layout(title = "xgboost HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_xgboost <- tune_xgboost %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_xgboost

# 采用最优超参数组合训练最终模型
set.seed(42)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(traindata)
final_xgboost

##################################################################

# 训练集预测评估
predtrain_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "Xgboost", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtrain_xgboost$proc)
pROC::ci.auc(predtrain_xgboost$proc)

# 预测评估测试集预测评估
predtest_xgboost <- eval4cls2(
  model = final_xgboost, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "Xgboost", 
  datasetname = "testdata",
  cutoff = predtrain_xgboost$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_xgboost$proc)
pROC::ci.auc(predtest_xgboost$proc)

# ROC比较检验
pROC::roc.test(predtrain_xgboost$proc, predtest_xgboost$proc)


# 合并训练集和测试集上ROC曲线
predtrain_xgboost$rocresult %>%
  bind_rows(predtest_xgboost$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_xgboost$prresult %>%
  bind_rows(predtest_xgboost$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_xgboost$caliresult %>%
  bind_rows(predtest_xgboost$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_xgboost$metrics %>%
  bind_rows(predtest_xgboost$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_xgboost <- bestcv4cls2(
  wkflow = wk_xgboost,
  tuneresult = tune_xgboost,
  hpbest = hpbest_xgboost,
  yname = "AHD",
  modelname = "Xgboost",
  v = 5,
  positivelevel = yourpositivelevel
)
# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_xgboost2 <- final_xgboost %>%
  extract_fit_engine()
final_xgboost2

# 变量重要性
importance_matrix <- 
  xgboost::xgb.importance(model = final_xgboost2)
print(importance_matrix)
xgboost::xgb.plot.importance(
  importance_matrix = importance_matrix,
  measure = "Cover",
  col = "skyblue",
  family = "serif"
)


######################## DALEX解释对象

explainer_xgboost <- DALEXtra::explain_tidymodels(
  final_xgboost, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Xgboost"
)
# 变量重要性
vip_xgboost <- viplot(explainer_xgboost)
vipdata_xgboost <- vip_xgboost$data
vip_xgboost$plot



######################################################################

# 保存评估结果
save(datarecipe_xgboost,
     model_xgboost,
     wk_xgboost,
     #hpgrid_xgboost,  # 如果采用贝叶斯优化则替换为 param_xgboost
     tune_xgboost,
     predtrain_xgboost,
     predtest_xgboost,
     evalcv_xgboost,
     vipdata_xgboost,
     file = ".\\cls2\\evalresult_xgboost.RData")

###lightgbm
# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---lightgbm
# https://parsnip.tidymodels.org/reference/details_boost_tree_lightgbm.html

##############################################################
# 数据预处理配方
library(bonsai)
datarecipe_lightgbm <- recipe(formula = AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors())
datarecipe_lightgbm


# 设定模型
model_lightgbm <- boost_tree(
  mode = "classification",
  engine = "lightgbm",
  tree_depth = tune(),
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune()
)
model_lightgbm

# workflow
wk_lightgbm <- 
  workflow() %>%
  add_recipe(datarecipe_lightgbm) %>%
  add_model(model_lightgbm)
wk_lightgbm

##############################################################
#########################  超参数寻优2选1-贝叶斯优化

# 更新超参数范围
param_lightgbm <- model_lightgbm %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)),
         min_n = min_n(c(15, 55)))

# 贝叶斯优化超参数
set.seed(42)
tune_lightgbm <- wk_lightgbm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_lightgbm,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束

# 交叉验证结果
eval_tune_lightgbm <- tune_lightgbm %>%
  collect_metrics()
eval_tune_lightgbm

# 图示
# autoplot(tune_lightgbm)
eval_tune_lightgbm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mtry', values = ~mtry),
      list(label = 'trees', values = ~trees),
      list(label = 'min_n', values = ~min_n),
      list(label = 'tree_depth', values = ~tree_depth),
      list(label = 'learn_rate', values = ~learn_rate),
      list(label = 'loss_reduction', values = ~loss_reduction)
    )
  ) %>%
  plotly::layout(title = "lightgbm HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_lightgbm <- tune_lightgbm %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_lightgbm

# 采用最优超参数组合训练最终模型
set.seed(42)
final_lightgbm <- wk_lightgbm %>%
  finalize_workflow(hpbest_lightgbm) %>%
  fit(traindata)
final_lightgbm

##################################################################

# 训练集预测评估
predtrain_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "Lightgbm", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)

# pROC包auc值及其置信区间
pROC::auc(predtrain_lightgbm$proc)
pROC::ci.auc(predtrain_lightgbm$proc)

# 预测评估测试集预测评估
predtest_lightgbm <- eval4cls2(
  model = final_lightgbm, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "Lightgbm", 
  datasetname = "testdata",
  cutoff = predtrain_lightgbm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_lightgbm$proc)
pROC::ci.auc(predtest_lightgbm$proc)

# ROC比较检验
pROC::roc.test(predtrain_lightgbm$proc, predtest_lightgbm$proc)


# 合并训练集和测试集上ROC曲线
predtrain_lightgbm$rocresult %>%
  bind_rows(predtest_lightgbm$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_lightgbm$prresult %>%
  bind_rows(predtest_lightgbm$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_lightgbm$caliresult %>%
  bind_rows(predtest_lightgbm$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_lightgbm$metrics %>%
  bind_rows(predtest_lightgbm$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_lightgbm <- bestcv4cls2(
  wkflow = wk_lightgbm,
  tuneresult = tune_lightgbm,
  hpbest = hpbest_lightgbm,
  yname = "AHD",
  modelname = "Lightgbm",
  v = 5,
  positivelevel = yourpositivelevel
)


# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_lightgbm2 <- final_lightgbm %>%
  extract_fit_engine()
final_lightgbm2

# 保存lightgbm模型比较特殊
model_file <- 
  tempfile(pattern = "Lightgbm", tmpdir = ".", fileext = ".txt")
lightgbm::lgb.save(final_lightgbm2, model_file)

# # 加载也需要自己的函数
# load_booster <- lightgbm::lgb.load(file.choose())

# 变量重要性
lightgbm::lgb.importance(final_lightgbm2, percentage = T)
lightgbm::lgb.plot.importance(
  lightgbm::lgb.importance(final_lightgbm2, percentage = T)
)

# 变量对预测的贡献
lightgbm::lgb.interprete(
  final_lightgbm2, 
  as.matrix(final_lightgbm %>%
              extract_recipe() %>%
              bake(new_data = traindata) %>%
              dplyr::select(-AHD)), 
  1:2
)
lightgbm::lgb.plot.interpretation(
  lightgbm::lgb.interprete(
    final_lightgbm2, 
    as.matrix(final_lightgbm %>%
                extract_recipe() %>%
                bake(new_data = traindata) %>%
                dplyr::select(-AHD)),
    1:2
  )[[1]]
)

######################## DALEX解释对象

explainer_lightgbm <- DALEXtra::explain_tidymodels(
  final_lightgbm, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Lightgbm"
)
# 变量重要性
vip_lightgbm <- viplot(explainer_lightgbm)
vipdata_lightgbm <- vip_lightgbm$data
vip_lightgbm$plot





######################################################################

# 保存评估结果
save(datarecipe_lightgbm,
     model_lightgbm,
     wk_lightgbm,
     #hpgrid_lightgbm, # 如果采用贝叶斯优化则替换为 param_lightgbm
     tune_lightgbm,
     predtrain_lightgbm,
     predtest_lightgbm,
     evalcv_lightgbm,
     vipdata_lightgbm,
     file = ".\\cls2\\evalresult_lightgbm.RData")

###svm

##############################################################


# 数据预处理配方
datarecipe_svm <- recipe(formula = AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())
datarecipe_svm


# 设定模型
model_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_args(class.weights = c("No" = 1, "Yes" = 2)) 
# 此处NoYes是因变量的取值水平，换数据之后要相应更改
# 后面的数字12表示分类错误时的成本权重，无需设定时都等于1即可
model_svm

# workflow
wk_svm <- 
  workflow() %>%
  add_recipe(datarecipe_svm) %>%
  add_model(model_svm)
wk_svm

##############################################################
############################  超参数寻优2选1-网格搜索


#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_svm <- wk_svm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_svm <- tune_svm %>%
  collect_metrics()
eval_tune_svm

# 图示
# autoplot(tune_svm)
eval_tune_svm %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'cost', values = ~cost),
      list(label = 'rbf_sigma', values = ~rbf_sigma,
           font = list(family = "serif"))
    )
  ) %>%
  plotly::layout(title = "SVM HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_svm <- tune_svm %>%
  select_best(metric = "roc_auc")
hpbest_svm

# 采用最优超参数组合训练最终模型
set.seed(42)
final_svm <- wk_svm %>%
  finalize_workflow(hpbest_svm) %>%
  fit(traindata)
final_svm

##################################################################

# 训练集预测评估
predtrain_svm <- eval4cls2(
  model = final_svm, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "SVM", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtrain_svm$proc)
pROC::ci.auc(predtrain_svm$proc)

# 预测评估测试集预测评估
predtest_svm <- eval4cls2(
  model = final_svm, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "SVM", 
  datasetname = "testdata",
  cutoff = predtrain_svm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_svm$proc)
pROC::ci.auc(predtest_svm$proc)

# ROC比较检验
pROC::roc.test(predtrain_svm$proc, predtest_svm$proc)


# 合并训练集和测试集上ROC曲线
predtrain_svm$rocresult %>%
  bind_rows(predtest_svm$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_svm$prresult %>%
  bind_rows(predtest_svm$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_svm$caliresult %>%
  bind_rows(predtest_svm$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_svm$metrics %>%
  bind_rows(predtest_svm$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_svm <- bestcv4cls2(
  wkflow = wk_svm,
  tuneresult = tune_svm,
  hpbest = hpbest_svm,
  yname = "AHD",
  modelname = "SVM",
  v = 5,
  positivelevel = yourpositivelevel
)




# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_svm2 <- final_svm %>%
  extract_fit_engine()
final_svm2


######################## DALEX解释对象

explainer_svm <- DALEXtra::explain_tidymodels(
  final_svm, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "SVM"
)
# 变量重要性
vip_svm <- viplot(explainer_svm)
vipdata_svm <- vip_svm$data
vip_svm$plot




######################################################################

# 保存评估结果
save(datarecipe_svm,
     model_svm,
     wk_svm,
     #hpgrid_svm, # 如果采用贝叶斯优化则删除这一行
     tune_svm,
     predtrain_svm,
     predtest_svm,
     evalcv_svm,
     vipdata_svm,
     file = ".\\cls2\\evalresult_svm.RData")

###单隐藏神经网络

##############################################################



# 数据预处理配方
datarecipe_mlp <- recipe(formula = AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>% 
  step_range(all_predictors())
datarecipe_mlp


# 设定模型
model_mlp <- mlp(
  mode = "classification",
  engine = "nnet",
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_args(MaxNWts = 10000)
model_mlp

# workflow
wk_mlp <- 
  workflow() %>%
  add_recipe(datarecipe_mlp) %>%
  add_model(model_mlp)
wk_mlp

##############################################################
############################  超参数寻优2选1-网格搜索
#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_mlp <- wk_mlp %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_mlp <- tune_mlp %>%
  collect_metrics()
eval_tune_mlp

# 图示
# autoplot(tune_mlp)
eval_tune_mlp %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'hidden_units', values = ~hidden_units),
      list(label = 'penalty', values = ~penalty),
      list(label = 'epochs', values = ~epochs)
    )
  ) %>%
  plotly::layout(title = "MLP HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_mlp <- tune_mlp %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_mlp

# 采用最优超参数组合训练最终模型
set.seed(42)
final_mlp <- wk_mlp %>%
  finalize_workflow(hpbest_mlp) %>%
  fit(traindata)
final_mlp

##################################################################

# 训练集预测评估
predtrain_mlp <- eval4cls2(
  model = final_mlp, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "MLP", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtrain_mlp$proc)
pROC::ci.auc(predtrain_mlp$proc)

# 预测评估测试集预测评估
predtest_mlp <- eval4cls2(
  model = final_mlp, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "MLP", 
  datasetname = "testdata",
  cutoff = predtrain_mlp$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_mlp$proc)
pROC::ci.auc(predtest_mlp$proc)

# ROC比较检验
pROC::roc.test(predtrain_mlp$proc, predtest_mlp$proc)


# 合并训练集和测试集上ROC曲线
predtrain_mlp$rocresult %>%
  bind_rows(predtest_mlp$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_mlp$prresult %>%
  bind_rows(predtest_mlp$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_mlp$caliresult %>%
  bind_rows(predtest_mlp$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_mlp$metrics %>%
  bind_rows(predtest_mlp$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_mlp <- bestcv4cls2(
  wkflow = wk_mlp,
  tuneresult = tune_mlp,
  hpbest = hpbest_mlp,
  yname = "AHD",
  modelname = "MLP",
  v = 5,
  positivelevel = yourpositivelevel
)





# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_mlp2 <- final_mlp %>%
  extract_fit_engine()
final_mlp2

######################## DALEX解释对象

explainer_mlp <- DALEXtra::explain_tidymodels(
  final_mlp, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "MLP"
)
# 变量重要性
vip_mlp <- viplot(explainer_mlp)
vipdata_mlp <- vip_mlp$data
vip_mlp$plot




######################################################################

# 保存评估结果
save(datarecipe_mlp,
     model_mlp,
     wk_mlp,
     #hpgrid_mlp, # 如果采用贝叶斯优化则删除这一行
     tune_mlp,
     predtrain_mlp,
     predtest_mlp,
     evalcv_mlp,
     vipdata_mlp,
     file = ".\\cls2\\evalresult_mlp.RData")


###knn
# 模型机器---R语言tidymodels包机器学习分类与回归模型---二分类---KNN
# https://parsnip.tidymodels.org/reference/details_nearest_neighbor_kknn.html

##############################################################



# 数据预处理配方
datarecipe_knn <- recipe(formula = AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())
datarecipe_knn


# 设定模型
model_knn <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  
  neighbors = tune(),
  weight_func = tune(),
  dist_power = 2
)
model_knn

# workflow
wk_knn <- 
  workflow() %>%
  add_recipe(datarecipe_knn) %>%
  add_model(model_knn)
wk_knn

##############################################################
############################  超参数寻优2选1-网格搜索


#########################  超参数寻优2选1-贝叶斯优化

# 更新超参数范围
param_knn <- model_knn %>%
  extract_parameter_set_dials() %>%
  update(neighbors = neighbors(c(5, 35)),
         weight_func = weight_func(c("rectangular",  "triangular")))

# 贝叶斯优化超参数
set.seed(42)
tune_knn <- wk_knn %>%
  tune_bayes(
    resamples = folds,
    initial = 20,
    iter = 50,
    param_info = param_knn,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_knn <- tune_knn %>%
  collect_metrics()
eval_tune_knn

# 图示
# autoplot(tune_knn)
eval_tune_knn %>% 
  filter(.metric == "roc_auc") %>%
  mutate(weight_func2 = as.numeric(as.factor(weight_func))) %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'neighbors', values = ~neighbors),
      list(label = 'weight_func', values = ~weight_func2,
           range = c(1,length(unique(eval_tune_knn$weight_func))), 
           tickvals = 1:length(unique(eval_tune_knn$weight_func)),
           ticktext = sort(unique(eval_tune_knn$weight_func)))
    )
  ) %>%
  plotly::layout(title = "KNN HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_knn <- tune_knn %>%
  select_by_one_std_err(metric = "roc_auc", desc(neighbors))
hpbest_knn

# 采用最优超参数组合训练最终模型
set.seed(42)
final_knn <- wk_knn %>%
  finalize_workflow(hpbest_knn) %>%
  fit(traindata)
final_knn

##################################################################

# 训练集预测评估
predtrain_knn <- eval4cls2(
  model = final_knn, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "KNN", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtrain_knn$proc)
pROC::ci.auc(predtrain_knn$proc)

# 预测评估测试集预测评估
predtest_knn <- eval4cls2(
  model = final_knn, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "KNN", 
  datasetname = "testdata",
  cutoff = predtrain_knn$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_knn$proc)
pROC::ci.auc(predtest_knn$proc)

# ROC比较检验
pROC::roc.test(predtrain_knn$proc, predtest_knn$proc)


# 合并训练集和测试集上ROC曲线
predtrain_knn$rocresult %>%
  bind_rows(predtest_knn$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_knn$prresult %>%
  bind_rows(predtest_knn$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_knn$caliresult %>%
  bind_rows(predtest_knn$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_knn$metrics %>%
  bind_rows(predtest_knn$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_knn <- bestcv4cls2(
  wkflow = wk_knn,
  tuneresult = tune_knn,
  hpbest = hpbest_knn,
  yname = "AHD",
  modelname = "KNN",
  v = 5,
  positivelevel = yourpositivelevel
)




# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_knn2 <- final_knn %>%
  extract_fit_engine()
final_knn2


######################## DALEX解释对象

explainer_knn <- DALEXtra::explain_tidymodels(
  final_knn, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "KNN"
)
# 变量重要性
vip_knn <- viplot(explainer_knn)
vipdata_knn <- vip_knn$data
vip_knn$plot



# 保存评估结果
save(datarecipe_knn,
     model_knn,
     wk_knn,
     #hpgrid_knn, # 如果采用贝叶斯优化则删掉这一行
     tune_knn,
     predtrain_knn,
     predtest_knn,
     evalcv_knn,
     vipdata_knn,
     file = ".\\cls2\\evalresult_knn.RData")

###逻辑回归

##############################################################



# 数据预处理配方
datarecipe_logistic <- recipe(AHD ~ ., traindata)
datarecipe_logistic


# 设定模型
model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)
model_logistic

# workflow
wk_logistic <- 
  workflow() %>%
  add_recipe(datarecipe_logistic) %>%
  add_model(model_logistic)
wk_logistic

# 训练模型
set.seed(42)
final_logistic <- wk_logistic %>%
  fit(traindata)
final_logistic

##################################################################

# 训练集预测评估
predtrain_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "Logistic", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtrain_logistic$proc)
pROC::ci.auc(predtrain_logistic$proc)

# 预测评估测试集预测评估
predtest_logistic <- eval4cls2(
  model = final_logistic, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "Logistic", 
  datasetname = "testdata",
  cutoff = predtrain_logistic$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_logistic$proc)
pROC::ci.auc(predtest_logistic$proc)

# ROC比较检验
pROC::roc.test(predtrain_logistic$proc, predtest_logistic$proc)


# 合并训练集和测试集上ROC曲线
predtrain_logistic$rocresult %>%
  bind_rows(predtest_logistic$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_logistic$prresult %>%
  bind_rows(predtest_logistic$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_logistic$caliresult %>%
  bind_rows(predtest_logistic$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_logistic$metrics %>%
  bind_rows(predtest_logistic$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

##################################################################

# 交叉验证
set.seed(42)
cv_logistic <- 
  wk_logistic %>%
  fit_resamples(
    folds,
    metrics = metricset_cls2,
    control = control_resamples(save_pred = T,
                                verbose = T,
                                event_level = "second",
                                parallel_over = "everything",
                                save_workflow = T)
  )
cv_logistic

# 交叉验证指标结果
evalcv_logistic <- list()
# 评估指标设定
metrictemp <- metric_set(yardstick::roc_auc, yardstick::pr_auc)
evalcv_logistic$evalcv <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  metrictemp(AHD, .pred_Yes, event_level = "second") %>%
  group_by(.metric) %>%
  mutate(model = "logistic",
         mean = mean(.estimate),
         sd = sd(.estimate)/sqrt(length(folds$splits)))
evalcv_logistic$evalcv

# 交叉验证预测结果图示
# ROC
evalcv_logistic$cvroc <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(AHD, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic$evalcv %>% filter(.metric == "roc_auc"), 
            by = "id") %>%
  mutate(idAUC = paste(id, " ROCAUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))
evalcv_logistic$cvroc

# PR
evalcv_logistic$cvpr <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  pr_curve(AHD, .pred_Yes, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic$evalcv %>% filter(.metric == "pr_auc"), 
            by = "id") %>%
  mutate(idAUC = paste(id, " PRAUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = recall, y = precision, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", intercept = 1, slope = -1) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))
evalcv_logistic$cvpr



###################################################################

# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_logistic2 <- final_logistic %>%
  extract_fit_engine()
final_logistic2


######################## DALEX解释对象

explainer_logistic <- DALEXtra::explain_tidymodels(
  final_logistic, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Logistic"
)
# 变量重要性
vip_logistic <- viplot(explainer_logistic)
vipdata_logistic <- vip_logistic$data
vip_logistic$plot



# 保存评估结果
save(datarecipe_logistic,
     model_logistic,
     wk_logistic,
     cv_logistic,
     predtrain_logistic,
     predtest_logistic,
     evalcv_logistic,
     vipdata_logistic,
     file = ".\\cls2\\evalresult_logistic.RData")



###弹性网络

##############################################################



# 数据预处理配方
datarecipe_enet <- recipe(formula = AHD ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = new_dummy_names) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())
datarecipe_enet


# 设定模型
model_enet <- logistic_reg(
  mode = "classification",
  engine = "glmnet",
  mixture = tune(),
  penalty = tune()
)
model_enet

# workflow
wk_enet <- 
  workflow() %>%
  add_recipe(datarecipe_enet) %>%
  add_model(model_enet)
wk_enet

##############################################################
############################   超参数寻优2选1-网格搜索
#########################  超参数寻优2选1-贝叶斯优化

# 贝叶斯优化超参数
set.seed(42)
tune_enet <- wk_enet %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束


# 交叉验证结果
eval_tune_enet <- tune_enet %>%
  collect_metrics()
eval_tune_enet

# 图示
# autoplot(tune_enet)
eval_tune_enet %>% 
  filter(.metric == "roc_auc") %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'mixture', values = ~mixture),
      list(label = 'penalty', values = ~penalty)
    )
  ) %>%
  plotly::layout(title = "ENet HPO Guided by AUCROC",
                 font = list(family = "serif"))

# 经过交叉验证得到的最优超参数
hpbest_enet <- tune_enet %>%
  select_by_one_std_err(metric = "roc_auc", desc(penalty))
hpbest_enet

# 采用最优超参数组合训练最终模型
set.seed(42)
final_enet <- wk_enet %>%
  finalize_workflow(hpbest_enet) %>%
  fit(traindata)
final_enet

##################################################################

# 训练集预测评估
predtrain_enet <- eval4cls2(
  model = final_enet, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "ENet", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
# pROC包auc值及其置信区间
pROC::auc(predtrain_enet$proc)
pROC::ci.auc(predtrain_enet$proc)

# 预测评估测试集预测评估
predtest_enet <- eval4cls2(
  model = final_enet, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "ENet", 
  datasetname = "testdata",
  cutoff = predtrain_enet$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)


# pROC包auc值及其置信区间
pROC::auc(predtest_enet$proc)
pROC::ci.auc(predtest_enet$proc)

# ROC比较检验
pROC::roc.test(predtrain_enet$proc, predtest_enet$proc)


# 合并训练集和测试集上ROC曲线
predtrain_enet$rocresult %>%
  bind_rows(predtest_enet$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_enet$prresult %>%
  bind_rows(predtest_enet$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_enet$caliresult %>%
  bind_rows(predtest_enet$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_enet$metrics %>%
  bind_rows(predtest_enet$metrics) %>%
  dplyr::select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 最优超参数交叉验证的结果
evalcv_enet <- bestcv4cls2(
  wkflow = wk_enet,
  tuneresult = tune_enet,
  hpbest = hpbest_enet,
  yname = "AHD",
  modelname = "ENet",
  v = 5,
  positivelevel = yourpositivelevel
)

# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

# 提取最终的算法模型
final_enet2 <- final_enet %>%
  extract_fit_engine()
final_enet2

# 非零系数自变量
tidy(final_enet) %>%
  filter(term != "(Intercept)", estimate != 0) %>%
  pull(term)

######################## DALEX解释对象

explainer_enet <- DALEXtra::explain_tidymodels(
  final_enet, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "ENet"
)
# 变量重要性
vip_enet <- viplot(explainer_enet)
vipdata_enet <- vip_enet$data
vip_enet$plot


######################################################################

# 保存评估结果
save(datarecipe_enet,
     model_enet,
     wk_enet,
     #hpgrid_enet,   # 如果采用贝叶斯优化则删掉这一行
     tune_enet,
     predtrain_enet,
     predtest_enet,
     evalcv_enet,
     vipdata_enet,
     file = ".\\cls2\\evalresult_enet.RData")

###堆叠模型
# 数据预处理配方
datarecipe_dt <- recipe(formula = AHD ~ ., traindata)
datarecipe_dt


# 用于构建stacking模型的样本自变量值是候选基础模型的样本外预测结果
library(bonsai)
library(stacks)

##############################

# 也可以是之前单个模型建模的结果
load(".\\cls2\\evalresult_svm.RData")
load(".\\cls2\\evalresult_xgboost.RData")
load(".\\cls2\\evalresult_lightgbm.RData")
models_stack <- 
  stacks() %>% 
  add_candidates(tune_svm) %>%
  add_candidates(tune_xgboost) %>%
  add_candidates(tune_lightgbm) 

models_stack

##############################
###堆叠改
library(tidymodels)
library(stacks)

set.seed(42)

# 创建参数网格，包括 alpha 和 penalty
alpha_values <- seq(0, 1, length = 10)  # 从岭回归到 LASSO的范围
penalty_values <- 10^seq(-3, 1, length = 20)  # penalty值范围

# 调用 blend_predictions，同时调优 alpha 和 penalty
meta_stack <- blend_predictions(
  models_stack, 
  penalty = penalty_values, 
  control = control_grid(
    save_pred = TRUE, 
    verbose = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  ),
  mixture = alpha_values  # 同时调优 alpha 参数
)

# 输出调优结果
print(meta_stack)

# 可视化不同 alpha 和 penalty 对模型性能的影响
autoplot(meta_stack) +
  theme_bw()
# 拟合选定的基础模型
set.seed(42)
final_stack <- fit_members(meta_stack)
final_stack

######################################################

# 应用stacking模型预测并评估
# 训练集
predtrain_stack <- eval4cls2(
  model = final_stack, 
  dataset = traindata, 
  yname = "AHD", 
  modelname = "Stacking", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_stack$prediction
predtrain_stack$predprobplot
predtrain_stack$rocplot
predtrain_stack$prplot
predtrain_stack$caliplot
predtrain_stack$cmplot
predtrain_stack$metrics
predtrain_stack$diycutoff
predtrain_stack$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_stack$proc)
pROC::ci.auc(predtrain_stack$proc)

# 测试集
predtest_stack <- eval4cls2(
  model = final_stack, 
  dataset = testdata, 
  yname = "AHD", 
  modelname = "Stacking", 
  datasetname = "testdata",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_stack$prediction
predtest_stack$predprobplot
predtest_stack$rocplot
predtest_stack$prplot
predtest_stack$caliplot
predtest_stack$cmplot
predtest_stack$metrics
predtest_stack$diycutoff
predtest_stack$ksplot

# pROC包auc值及其置信区间
pROC::auc(predtest_stack$proc)
pROC::ci.auc(predtest_stack$proc)

# ROC比较检验
pROC::roc.test(predtrain_stack$proc, predtest_stack$proc)


# 合并训练集和测试集上ROC曲线
predtrain_stack$rocresult %>%
  bind_rows(predtest_stack$rocresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上PR曲线
predtrain_stack$prresult %>%
  bind_rows(predtest_stack$prresult) %>%
  mutate(dataAUC = paste(data, curvelab),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1), 
                     breaks = seq(0, 1, by = 0.2),
                     labels = seq(0, 1, by = 0.2)) +
  labs(color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上校准曲线
predtrain_stack$caliresult %>%
  bind_rows(predtest_stack$caliresult) %>%
  mutate(data = forcats::as_factor(data)) %>%
  ggplot(aes(x = predprobgroup,
             y = Fraction, 
             color = data)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, pch = 15) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1),
                     breaks = seq(0, 1, by = 0.1),
                     labels = seq(0, 1, by = 0.1)) +
  labs(x = "Bin Midpoint", y = "Event Rate", color = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))

# 合并训练集和测试集上性能指标
predtrain_stack$metrics %>%
  bind_rows(predtest_stack$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)



# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-AHD)
colnames(traindatax)

# 分类型、连续型自变量名称
catvars <- getcategory(traindatax)
convars <- getcontinuous(traindatax)

######################## DALEX解释对象

explainer_stack <- DALEXtra::explain_tidymodels(
  final_stack, 
  data = traindatax,
  y = ifelse(traindata$AHD == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Stacking"
)
# 变量重要性
vip_stack <- viplot(explainer_stack)
vipdata_stack <- vip_stack$data
vip_stack$plot

# 变量偏依赖图
pdplot(explainer_stack, convars)
pdplot(explainer_stack, "Age")
pdplot(explainer_stack, catvars)
pdplot(explainer_stack, "Thal")

###################################### iml解释对象

predictor_stack <- iml::Predictor$new(
  final_stack, 
  data = traindatax,
  y = traindata$AHD,
  predict.function = function(model, newdata){
    predict(model, newdata, type = "prob") %>%
      rename_with(~gsub(".pred_", "", .x))
  },
  type = "prob"
)
# 交互作用
interact_stack <- iml::Interaction$new(predictor_stack)
plot(interact_stack) +
  theme_minimal()

interact_stack_1vo <- 
  iml::Interaction$new(predictor_stack, feature = "Age")
plot(interact_stack_1vo) +
  theme_minimal()

interact_stack_1v1 <- iml::FeatureEffect$new(
  predictor_stack, 
  feature = c("Age", "Oldpeak"),
  method = "pdp"
)
plot(interact_stack_1v1) +
  scale_fill_viridis_c() +
  labs(fill = "") +
  theme_minimal()

###################################### lime单样本预测分解

explainer_stack <- lime::lime(
  traindatax,
  lime::as_classifier(final_stack, c(yournegativelevel, yourpositivelevel))
)
explanation_stack <- lime::explain(
  traindatax[1,],  # 训练集第1个样本
  explainer_stack, 
  n_labels = 2, 
  n_features = ncol(traindatax)
)
lime::plot_features(explanation_stack)

######################## fastshap包

shapresult_stack <- shap4cls2(
  finalmodel = final_stack,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") %>%
      dplyr::select(ends_with(yourpositivelevel)) %>%
      pull()
  },
  datax = traindatax,
  datay = traindata$AHD,
  yname = "AHD",
  flname = catvars,
  lxname = convars
)

# 基于shap的变量重要性
shapresult_stack$shapvipplot

# 单样本预测分解
shap41 <- shapviz::shapviz(
  shapresult_stack$shapley,
  X = traindatax
)
shapviz::sv_force(shap41, row_id = 1)  +  # 训练集第1个样本
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shap41, row_id = 1)  +  # 训练集第1个样本
  theme(text = element_text(family = "serif"))

# 所有分类变量的shap图示
shapresult_stack$shapplotd_facet
shapresult_stack$shapplotd_one
# 所有连续变量的shap图示
shapresult_stack$shapplotc_facet
shapresult_stack$shapplotc_one
shapresult_stack$shapplotc_one2
# 单变量shap图示
sdplot(shapresult_stack, "Thal", "AHD")
sdplot(shapresult_stack, "Age", "AHD")

# 所有变量一张图
# shap变量重要性
shapresult_stack$shapvipplot_unity
# shap依赖图
shapresult_stack$shapplot_unity

######################################################################

# 保存评估结果
save(predtrain_stack,
     predtest_stack,
     vipdata_stack,
     file = ".\\cls2\\evalresult_stack.RData")

