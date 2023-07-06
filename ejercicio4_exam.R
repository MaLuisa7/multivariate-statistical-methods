
datos_p4 <- read.csv("C:/Users/Maria Luisa/Downloads/datos_p4ex.csv", header = T)
library(mvtnorm)
library(ICSNP)
## prueba de comparaci?n de dos vectores Miu's ASUMIENDO NORMALIDAD EN LOS DATOS
nor  <- datos_p4[1:35,2:3] # turno 1
sur <- datos_p4[36:61,2:3] # turno 2

HotellingsT2(nor, sur, mu = rep(0,ncol(nor)), test = "f")  # supuesto de normalidad
# Hotelling's two sample T2-test
# H0 : mu1 = mu2
# H1 : mu1 != mu2
# data:  nor and sur
# T.2 = 26.106, df1 = 2, df2 = 58, p-value = 8.217e-09
# alternative hypothesis: true location difference is not equal to c(0,0,0,0)
# CONCLUSION :
# Se conoce que se rechaza H0 si pvalor es menor a alfa,

# pvalor es menor a alfa,n se rechaza H0, es decir si hay diferencia entre las medias de 
#dureza del agua y mortalidad entre las regiones

# se supone normalidad en los datos, por tanto se realiza la prueba


##prueba de Normal Multivariada
library(MVN)
mvn(nor, mvnTest = "hz", univariateTest = "AD")
# H0 : Los datos son normales multivariados
# h1 : Los datos no son normales multivariados
# Test               HZ      p value      MVN
# Henze-Zirkler 1.307551 0.002257558  NO
# Se rechaza H0 si pvalor menor a alfa,
# com pvalor menor a alfa, se rechaza h0, los datos del norte no son normales multiv

mvn(sur, mvnTest = "hz", univariateTest = "AD")
# H0 : Los datos son normales multivariados
# h1 : Los datos no son normales multivariados
# Test               HZ      p value      MVN
#  Henze-Zirkler 0.3109231 0.7553233 YES
# Se rechaza H0 si pvalor menor a alfa,
# com pvalor mayor a alfa, no se rechaza h0, los datos del sur son normales multiv




res <- boxM(as.matrix(datos_p4[,2:3]), as.factor(datos_p4$location ))
res
#Box's M-test for Homogeneity of Covariance Matrices

#data:  as.matrix(datos_p4[, 2:3])
#Chi-Sq (approx.) = 5.7782, df = 3, p-value = 0.1229

# se conoce que se rechaza H0 si pvalor es menor a alfa
# como pvalor es mayor a alfa, no se rechaza h0, 
# las meuestras tienen covarianzas  iguales
 