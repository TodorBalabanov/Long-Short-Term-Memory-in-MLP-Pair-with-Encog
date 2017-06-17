con1 = read.csv("con1.csv", header = FALSE)
boxplot(con1, notch=FALSE, main="Training Convergence of Single MLP", xlab="Time", ylab="ANN Error") 

con2 = read.csv("con2.csv", header = FALSE)
boxplot(con2, notch=FALSE, main="Training Convergence of MLP Pair", xlab="Time", ylab="ANN Error") 

epo1 = read.csv("epo1.csv", header = FALSE)
boxplot(epo1, notch=FALSE, main="Training Epochs for Single MLP", xlab="Time", ylab="Epochs") 

epo2 = read.csv("epo2.csv", header = FALSE)
boxplot(epo2, notch=FALSE, main="Training Epochs for MLP Pair", xlab="Time", ylab="Epochs") 
