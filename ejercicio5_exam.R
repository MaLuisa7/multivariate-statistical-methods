datos_p5 <- read.csv("C:/Users/Maria Luisa/Downloads/datos_p5_exam.csv", header = T)
library(mvtnorm)
library(ICSNP)
inf_manova <- manova(as.matrix(datos_p5[,2:9])~as.factor(datos_p5$Region), data=datos_p5)
summary(inf_manova, test="Wilks") #Roy

#Df    Wilks approx F num Df den Df    Pr(>F)    
#as.factor(datos_p5$Region)   2 0.031702    324.3     16   1124 < 2.2e-16 ***
#  Residuals                  569                                              
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Se conoce que se rechaza H0 si pvalor es menor a alfa,
# como p valor es menor a alfa, se rechaza H0, al menos hay un par de medias diferentes
# lo cual quiere decir que si hay distinción en los aceites pertenecientes a las diferentes áreas dentro de cada 
# región

#Como se supone muestra grande no hace falta realizar la prueba de normalidad multivariada