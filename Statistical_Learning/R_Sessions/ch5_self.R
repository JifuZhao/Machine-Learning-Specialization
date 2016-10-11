load("5.R.RData")
fit = lm(y ~ X1 + X2, data=Xy)
summary(fit)

?matplot
matplot(Xy, type="l")

require(boot)

