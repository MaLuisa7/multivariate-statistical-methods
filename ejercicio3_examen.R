 
datos_p3 <- read.csv("C:/Users/Maria Luisa/OneDrive/Documentos/MasterDataScience/MEM/datos_problema3_exam.csv", header = T)
library(mvtnorm)
library(ICSNP)
## prueba de comparaci?n de dos vectores Miu's ASUMIENDO NORMALIDAD EN LOS DATOS
health <- datos_p3[1:45,2:5] # turno 1
inftech <- datos_p3[46:102,2:5] # turno 2

# SUPUESTO : MUESTRA GRANDE
#Dado que health tiene n = 45 y p = 4 , n-p = 45-4 = 41 mayor a 40 , se dice que es una muestra grande
#Dado que Information Technology tiene n = 57 y p = 4 , n-p = 57-4 = 53 mayor a 40 , se dice que es una muestra grande

HotellingsT2(health, inftech, mu = rep(0,ncol(health)), test = "chi") # muestra grande

#Hotelling's two sample T2-test

#data:  health and inftech
#T.2 = 7.0759, df = 4, p-value = 0.1319
#alternative hypothesis: true location difference is not equal to c(0,0,0,0)

# Se conoce que se rechaza H0 si pvalor es menor a alfa,
#Como pvalor = 0.1319 es mayor a alfa, se toma un alfa de 0.05, 
#por tanto no se rechaza H0,
# existe igualdad de medias 
# no hay diferencia de medias entre el sector de health care y el de information tecnolgies


