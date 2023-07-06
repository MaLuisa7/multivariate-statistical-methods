import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import imageio as io
import imageio.v2 as imageio
import os
from sklearn.decomposition import PCA
import skimage
from skimage import color
from skimage import io

#ubicacion de las carpetas con cada tipo de flor
path = "D:/Users/Maria Luisa/OneDrive/Documentos/MasterDataScience/MEM/data_flower/"

path_amarilla = "f_amarilla/"
path_blanca = "f_blanca/"
path_morada = "f_morada/"
path_rosa = "f_rosa/"

imgs_amarilla = os.listdir(path + path_amarilla)  # 41
imgs_blanca = os.listdir(path + path_blanca)  # 57
imgs_morada = os.listdir(path + path_morada)  # 39
imgs_rosa = os.listdir(path + path_rosa)  # 39

imgtest = imageio.imread(path + 'f_' + 'amarilla' + '/' + imgs_amarilla[0])

# Asegurarme que todas las imagenes tengan el mismo tama;o
'''def caract_img(archivos_color, nombre_color):
    path_color = 'f_' + nombre_color + '/'
    lst_caract_img = []
    for i in range(0, len(archivos_color)):
        img = imageio.imread(path + path_color + archivos_color[i])
        dim1, dim2, dim3 = list(np.shape(img))[0], list(np.shape(img))[1], list(np.shape(img))[2]

        lst_caract_img.append([archivos_color[i], nombre_color, dim1, dim2, dim3])
        data = pd.DataFrame(data=lst_caract_img, columns=['i', 'color', 'dim1', 'dim2', 'dim3'])
    return data
caract_img= pd.concat([caract_img(imgs_amarilla, 'amarilla'),caract_img(imgs_blanca, 'blanca'),
                       caract_img(imgs_morada, 'morada'),caract_img(imgs_rosa, 'rosa')],axis=0)
descr_car_img = caract_img.groupby('color').describe()
plt.imshow(img1) display an image'''

#Lectura de imagenes, redimensionamiento de matriz a vector y concatenacion
def lst_vector_color(archivos_color, nombre_color, id_color):
    path_color = 'f_' + nombre_color + '/'
    num_flat = 128 * 128 * 4
    lst_vectores = []
    lst_class = []
    for i in range(0, len(archivos_color)):
        img = imageio.imread(path + path_color + archivos_color[i])
        # img = io.imread(path + path_color + archivos_color[i] , as_gray=True)
        img_rshp = img.reshape(1, num_flat)[0]
        lst_vectores.append(img_rshp)
        array_vectores = np.array(lst_vectores)
        lst_class.append(id_color)
    return array_vectores, lst_class

array_vect_amarilla, id_amarilla = lst_vector_color(imgs_amarilla, 'amarilla', 0)
array_vect_blanca, id_blanca = lst_vector_color(imgs_blanca, 'blanca', 1)
array_vect_morada, id_morada = lst_vector_color(imgs_morada, 'morada', 2)
array_vect_rosa, id_rosa = lst_vector_color(imgs_rosa, 'rosa', 3)

# Ordenamos  la matriz de caracteristicas
x = np.concatenate((array_vect_amarilla, array_vect_blanca, array_vect_morada, array_vect_rosa),
                   axis=0)  # (160, 65536) n>p no se cumple, se tiene que reducir
y = np.array(id_amarilla + id_blanca + id_morada + id_rosa).reshape(160, 1)
xy = np.concatenate((x, y), axis=1)
x_std = (x - x.mean()) / (x.std())
n1, p1 = list(x_std.shape)[0], list(x_std.shape)[1]

test1 = pd.DataFrame(xy).add_prefix('pixel_').rename(columns={'pixel_65536': 'Clase'})

# the n_components of PCA must be lower than min(n_samples, n_features).
pca = PCA(n_components=60)
pca.fit(x_std)
X_proj = pca.fit_transform(x_std)
loading = pd.DataFrame(pca.components_.T).round(2)
componentes = pca.components_ #eigenvector (60, 65536)
componentes = pd.DataFrame(data= componentes,columns = list(test1.columns)[:-1]).round(2)
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))
eigen_valores = pd.DataFrame(pca.explained_variance_).round(2)
ratio_var_exp = pd.DataFrame(pca.explained_variance_ratio_)
cum_var_exp = pd.DataFrame( np.cumsum(pca.explained_variance_ratio_))
var_cum_normal = pd.concat([ratio_var_exp,cum_var_exp], axis = 1)







#plot varianza
'''plt.bar(range(len(ratio_var_exp)), ratio_var_exp, alpha=0.5, align='center',
            label='Varianza individual explicada')
plt.step(range(len(ratio_var_exp)), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
plt.ylabel('Ratio de Varianza Explicada')
plt.xlabel('Componentes Principales')
plt.legend(loc='best')
plt.tight_layout()'''




#Scatter plot en 3D
'''
x_ama = X_proj[:40,0]
y_ama = X_proj[:40,1]
x_bla = X_proj[40:80,0]
y_bla = X_proj[40:80,1]
x_mor = X_proj[80:120,0]
y_mor = X_proj[80:120,1]
x_ros = X_proj[120:,0]
y_ros = X_proj[120:,1]

plt.scatter(x_ama, y_ama, c='y', label= 'amarilla')
plt.scatter(x_bla, y_bla, c='k', label= 'blanca')
plt.scatter(x_mor, y_mor, c='m', label= 'morada')
plt.scatter(x_ros, y_ros, c='r', label= 'rosa')
plt.title("Diagrama de dispersión para dos componentes")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada')
plt.title("Varianza Explicada por ACP")
plt.ylim(0,1.2)
plt.show()
'''

#se calcula la inversa del pca
X_inv_proj = pca.inverse_transform(X_proj)
#se redimensiona
X_proj_img = np.reshape(X_inv_proj, (160, 128, 128, 4))
#SE elige una imagen
img1_recov = X_proj_img[121]
#Se des-estandariza
x_1 = (img1_recov * x.std()) + x.mean()

# Comparación de flores
'''plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(imageio.imread(path + 'f_' + 'rosa' + '/' + imgs_rosa[1]))
plt.title("Flor rosa original")
plt.subplot(1, 2, 2)
plt.imshow(x_1.astype('uint8'))
plt.title("Flor rosa con transformación de PCA")
plt.show()
plt.figure()'''

##Grafico en 3d
'''
from mpl_toolkits import mplot3d
plt.style.use('default')
# Plot scaled features
#amarillas
xdata_amar = X_proj[:40, 0]
ydata_amar = X_proj[:40, 1]
zdata_amar = X_proj[:40, 2]
#blancas
xdata_blanca = X_proj[40:80, 0]
ydata_blanca = X_proj[40:80, 1]
zdata_blanca = X_proj[40:80, 2]
#moradas
xdata_mor= X_proj[80:120, 0]
ydata_mor = X_proj[80:120, 1]
zdata_mor = X_proj[80:120, 2]
#rosas
xdata_rosa = X_proj[120:, 0]
ydata_rosa = X_proj[120:, 1]
zdata_rosa = X_proj[120:, 2]
# Prepare 3D graph
fig = plt.figure()
ax = plt.axes(projection='3d')
# Plot 3D plot
ax.scatter(xdata_amar, ydata_amar, zdata_amar, c='y', marker='*',label='Amarilla')
ax.scatter(xdata_blanca, ydata_blanca, zdata_blanca, c='k', marker='o',label='Blanca')
ax.scatter(xdata_mor, ydata_mor, zdata_mor, c='m', marker='x',label='Morada')
ax.scatter(xdata_rosa, ydata_rosa, zdata_rosa, c='r', marker='^',label='Rosa')

# Plot title of graph
plt.title("Diagrama de dispersión 3D para 3 componentes")
plt.legend()
# Plot x, y, z even ticks
ticks = np.linspace(-3, 3, num=5)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

# Plot x, y, z labels
ax.set_xlabel('PCA 1', rotation=150)
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3', rotation=60)
plt.show()'''

#########EDA

# Histogramas de pixeles de flores
'''sns.displot(x[-1])
plt.title("Distribución de una imagen de flor color rosa")'''

# desriptivos
x_pd_grp = pd.DataFrame(xy).groupby(128 * 128 * 4).max()
x_pd_grp.round(2)

########## No funciono por que era muy grande :(
'''x = x.T
mu = np.mean(x, axis =0)
s_cov = np.cov(x)
#calculamos distancia
distance = [((i-mu) * np.linalg.inv(s_cov) * (i-mu).T).item() for i in x]'''

