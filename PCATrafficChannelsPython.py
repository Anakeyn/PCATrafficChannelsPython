# -*- coding: utf-8 -*-
"""
Created on Tue Jun  13 09:50:35 2019

@author: Pierre
"""
#########################################################################
# PCATrafficChannelsPython
# Analyse en Composantes Principalkes du trafic selon les canaux
# Auteur : Pierre Rouarch 2019 - Licence GPL 3
# Données : Issues de l'API de Google Analytics - 
# Comme illustration Nous allons travailler sur les données du site 
# https://www.networking-morbihan.com 

#############################################################
# On démarre ici pour récupérer les bibliothèques utiles !!
#############################################################
#def main():   #on ne va pas utiliser le main car on reste dans Spyder
#Chargement des bibliothèques utiles (décommebter au besoin)
import numpy as np #pour les vecteurs et tableaux notamment
import matplotlib.pyplot as plt  #pour les graphiques
#import scipy as sp  #pour l'analyse statistique
import pandas as pd  #pour les Dataframes ou tableaux de données
import seaborn as sns #graphiques étendues
#import math #notamment pour sqrt()
#from datetime import timedelta
#from scipy import stats
#pip install scikit-misc  #pas d'install conda ???
from skmisc import loess  #pour methode Loess compatible avec stat_smooth
#conda install -c conda-forge plotnine
from plotnine import *  #pour ggplot like
#conda install -c conda-forge mizani 
from mizani.breaks import date_breaks  #pour personnaliser les dates affichées
from sklearn.decomposition import PCA  #pour pca
from sklearn.preprocessing import StandardScaler  #pour standardization


#Si besoin Changement du répertoire par défaut pour mettre les fichiers de sauvegarde
#dans le même répertoire que le script.
import os
print(os.getcwd())  #verif
#mon répertoire sur ma machine - nécessaire quand on fait tourner le programme 
#par morceaux dans Spyder.
#myPath = "C:/Users/Pierre/CHEMIN"
#os.chdir(myPath) #modification du path
#print(os.getcwd()) #verif


############################################################
# TRAFIC GLOBAL RECUPERATION DES DONNEES
############################################################

myDateToParse = ['date']  #pour parser la variable date en datetime sinon object
dfPageViews = pd.read_csv("dfPageViews.csv", sep=";", dtype={'Année':object}, parse_dates=myDateToParse)
#verifs
dfPageViews.info()  #72821 enregistrements 

#Recupération des sources
mySourcesChannel = pd.read_csv("mySourcesChannel.csv", sep=";")
mySourcesChannel.info()
dfPageViews.info()


#On vire les blancs pour faire  le merge on
dfPageViews['source'] = dfPageViews['source'].str.strip()
mySourcesChannel['source'] = mySourcesChannel['source'].str.strip()

dfPVChannel = pd.merge(dfPageViews, mySourcesChannel, on='source', how='left')
dfPVChannel.info()
#voyons ce que l'on a comme valeurs.
dfPVChannel['channel'].value_counts()
sorted(dfPVChannel['channel'].unique())

############################################################################
# #Préparation des données pour l'ACP
############################################################################

#creation de la dataframe PVDataForACP 
PVDataForACP = dfPVChannel[['pagePath', 'channel', 'pageviews']].copy() #nouveau dataframe avec que la date et les canaux
PVDataForACP.info()
PVDataForACP = PVDataForACP.groupby(['pagePath','channel']).count()

#
PVDataForACP.reset_index(inplace=True)
PVDataForACP.info()

#création d'une colonne par type de channel
PVDataForACP = PVDataForACP.pivot(index='pagePath',columns='channel',values='pageviews')

#Mettre des 0 à la place de NaN
PVDataForACP .fillna(0,inplace=True)
PVDataForACP.to_csv("PVDataForACP.csv", sep=";", index=False)  #on sauvegarde si besoin
PVDataForACP.info()  #description des données
PVDataForACP.describe() #résumé des données


##########################################################################
# ACP - Analyse en Composantes Principales pour le 
# trafic Global - Chaque observation est une page 
##########################################################################
X=PVDataForACP.values  #uniquement les valeurs dans une matrice.
scaler = StandardScaler() #instancie un objet StandardScaler
scaler.fit(X) #appliqué aux données
X_scaled = scaler.fit_transform(X) #données transformées 

pca = PCA(n_components=5) #instancie un objet PCA
pca.fit(X_scaled)  #appliqué aux données scaled

pca.components_.T
pca.explained_variance_
pca.explained_variance_ratio_   #en pourcentage
pca.explained_variance_ratio_[0]
#Préparation des données pour affichage
dfpca = pd.DataFrame(data = pca.explained_variance_ratio_
             , columns = ['Variance Expliquée'])
dfpca.index.name = 'Composantes'
dfpca.reset_index(inplace=True)
dfpca['Composantes'] +=1

#Graphique
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.barplot(x='Composantes', y= 'Variance Expliquée', data=dfpca )
ax.set(xlabel='Composantes', ylabel='% Variance Expliquée',
       title="La première composante comprend " + "{0:.2f}%".format(pca.explained_variance_ratio_[0]*100) + "de l'information")
fig.text(.9,-.05,"Screeplot du % de variance des composantes de l'ACP Normalisée \n Corrélation de Pearson pour les canaux - toutes les pages", fontsize=9, ha="right")
#plt.show()
fig.savefig("All-PCA-StandardScaler-Pearson-screeplot-channel.png", bbox_inches="tight", dpi=600)


##############
##nuage des individus et axes des variables
labels=PVDataForACP.columns.values

score= X_scaled[:,0:2]
coeff=np.transpose(pca.components_[0:2, :])
n = coeff.shape[0]

xs = score[:,0]
ys = score[:,1]  
#
scalex = 1.0/(xs.max() - xs.min())
scaley = 1.0/(ys.max() - ys.min())

#Graphique du nuage des pages et des axes des variables.
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.scatterplot(xs * scalex,ys * scaley, alpha=0.4) #
for i in range(n):
    ax.arrow(0, 0, coeff[i,0]*1, coeff[i,1]*1,color = 'r',alpha = 0.5, head_width=.03)
    ax.text(coeff[i,0]*1.15, coeff[i,1]*1.15 , labels[i], color = 'r', ha = 'center', va = 'center')
ax.set(xlabel='Composante 1', ylabel='Composante 2',
       title="Les variables sont toutes du même bord de l'axe principal 1 \n Sur l'axe 2 seul Webmail se détache")
ax.set_ylim((-0.7, 1.1))

fig.text(.9,-.05,"Nuage des pages et axes des variables Normalisées via StandardScaler \n Corrélation de Pearson pour les canaux - toutes les pages", fontsize=9, ha="right")
#plt.show()
fig.savefig("All-PCA-StandardScaler-Pearson-cloud-channel.png", bbox_inches="tight", dpi=600)



##########################################################################
# Pour le traffic de base
##########################################################################
#Relecture ############
myDateToParse = ['date']  #pour parser la variable date en datetime sinon object
dfBasePageViews = pd.read_csv("dfBasePageViews.csv", sep=";", dtype={'Année':object}, parse_dates=myDateToParse)
#verifs
dfBasePageViews.dtypes
dfBasePageViews.count()  #37615
dfBasePageViews.head(20)

#On vire les blancs pour faire  le merge on
dfBasePageViews['source'] = dfBasePageViews['source'].str.strip()
mySourcesChannel['source'] = mySourcesChannel['source'].str.strip()

#récuperation de la variable channel dans la dataframe principale par un left join.
dfBasePVChannel = pd.merge(dfBasePageViews, mySourcesChannel, on='source', how='left')
dfBasePVChannel.info()
#voyons ce que l'on a comme valeurs.
dfBasePVChannel['channel'].value_counts()
sorted(dfBasePVChannel['channel'].unique())


############################################################################
# #Préparation des données pour l'ACP

#creation de la dataframe BasePVDataForACP 
BasePVDataForACP = dfBasePVChannel[['pagePath', 'channel', 'pageviews']].copy() #nouveau dataframe avec que la date et les canaux
BasePVDataForACP.info()
BasePVDataForACP = BasePVDataForACP.groupby(['pagePath','channel']).count()

#
BasePVDataForACP.reset_index(inplace=True)
BasePVDataForACP.info()

#création d'une colonne par type de channel
BasePVDataForACP = BasePVDataForACP.pivot(index='pagePath',columns='channel',values='pageviews')

#Mettre des 0 à la place de NaN
BasePVDataForACP .fillna(0,inplace=True)
BasePVDataForACP.to_csv("BasePVDataForACP.csv", sep=";", index=False)  #on sauvegarde si besoin
BasePVDataForACP.info()  #description des données
BasePVDataForACP.describe() #résumé des données


##########################################################################
# ACP - Analyse en Composantes Principales pour le 
# trafic Direct Marketing - Chaque observation est une page 
##########################################################################
X=BasePVDataForACP.values  #uniquement les valeurs dans une matrice.

scaler = StandardScaler() #instancie un objet StandardScaler
X_scaled = scaler.fit_transform(X) #données transformées centrage-réduction
print(X_scaled)



pcaBase = PCA(n_components=5) #instancie un objet PCA


pcaBase.fit(X_scaled)  #appliqué aux données scaled

pcaBase.components_.T
pcaBase.explained_variance_
pcaBase.explained_variance_ratio_   #en pourcentage
pcaBase.explained_variance_ratio_[0]
#Préparation des données pour affichage
dfpcaBase = pd.DataFrame(data = pcaBase.explained_variance_ratio_
             , columns = ['Variance Expliquée'])
dfpcaBase.index.name = 'Composantes'
dfpcaBase.reset_index(inplace=True)
dfpcaBase['Composantes'] +=1

#Graphique
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.barplot(x='Composantes', y= 'Variance Expliquée', data=dfpcaBase )
#fig.suptitle("La première composante comprend déja " + "{0:.2f}%".format(pca.explained_variance_ratio_[0]*100) + "de l'information", fontsize=10, fontweight='bold')
ax.set(xlabel='Composantes', ylabel='% Variance Expliquée',
       title="La première composante comprend " + "{0:.2f}%".format(pcaBase.explained_variance_ratio_[0]*100) + "de l'information")
fig.text(.9,-.05,"Screeplot du % de variance des composantes de l'ACP Normalisée \n Corrélation de Pearson pour les canaux - pages de base", fontsize=9, ha="right")
#plt.show()
fig.savefig("Base-PCA-StandardScaler-Pearson-screeplot-channel.png", bbox_inches="tight", dpi=600)


##############
##nuage des individus et axes des variables
#Labels
labels=BasePVDataForACP.columns.values
score= X_scaled[:,0:2]
coeff=np.transpose(pcaBase.components_[0:2, :])
n = coeff.shape[0]

xs = score[:,0]
ys = score[:,1]  
#
scalex = 1.0/(xs.max() - xs.min())
scaley = 1.0/(ys.max() - ys.min())

#Graphique du nuage des pages et des axes des variables.
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.scatterplot(xs * scalex,ys * scaley, alpha=0.4) #
for i in range(n):
    ax.arrow(0, 0, coeff[i,0]*1, coeff[i,1]*1,color = 'r',alpha = 0.5, head_width=.03)
    ax.text(coeff[i,0]*1.15, coeff[i,1]*1.15 , labels[i], color = 'r', ha = 'center', va = 'center')
ax.set(xlabel='Composante 1', ylabel='Composante 2',
       title="Les variables sont toutes du même bord de l'axe principal 1 \n ")
ax.set_ylim((-0.7, 1.1))

fig.text(.9,-.05,"Nuage des pages et axes des variables Normalisées via StandardScaler \n Corrélation de Pearson pour les canaux - pages de bases", fontsize=9, ha="right")
#plt.show()
fig.savefig("Base-PCA-StandardScaler-Pearson-cloud-channel.png", bbox_inches="tight", dpi=600)


##########################################################################
#regardons pour le trafic Direct  Marketing uniquement i.e le traffic dont
# la source a dirigé vers une page Articles Marketing 
##########################################################################
#Relecture ############
myDateToParse = ['date']  #pour parser la variable date en datetime sinon object
dfDMPageViews = pd.read_csv("dfDMPageViews.csv", sep=";", dtype={'Année':object}, parse_dates=myDateToParse)
#verifs
dfDMPageViews.dtypes
dfDMPageViews.count()  #28553
dfDMPageViews.head(20)

#On vire les blancs pour faire  le merge on
dfDMPageViews['source'] = dfDMPageViews['source'].str.strip()
mySourcesChannel['source'] = mySourcesChannel['source'].str.strip()
#recuperation de la variable channel dans la dataframe principale par un left join.
dfDMPVChannel = pd.merge(dfDMPageViews, mySourcesChannel, on='source', how='left')
dfDMPVChannel.info()
#voyons ce que l'on a comme valeurs.
dfDMPVChannel['channel'].value_counts()
sorted(dfDMPVChannel['channel'].unique())

#Préparation des données pour l'ACP - Chaque observation est une page 

#creation de la dataframe DMPVDataForACP 
DMPVDataForACP = dfDMPVChannel[['pagePath', 'channel', 'pageviews']].copy() #nouveau dataframe avec que la date et les canaux
DMPVDataForACP.info()
DMPVDataForACP = DMPVDataForACP.groupby(['pagePath','channel']).count()

#
DMPVDataForACP.reset_index(inplace=True)
DMPVDataForACP.info()

#création d'une colonne par type de channel
DMPVDataForACP = DMPVDataForACP.pivot(index='pagePath',columns='channel',values='pageviews')

#Mettre des 0 à la place de NaN
DMPVDataForACP .fillna(0,inplace=True)
DMPVDataForACP.to_csv("DMPVDataForACP.csv", sep=";", index=False)  #on sauvegarde si besoin
DMPVDataForACP.info()  #description des données
DMPVDataForACP.describe() #résumé des données


##########################################################################
# ACP - Analyse en Composantes Principales pour le 
# trafic Direct Marketing - Chaque observation est une page 
##########################################################################
from sklearn.decomposition import PCA

X=DMPVDataForACP.values  #uniquement les valeurs dans une matrice.

from sklearn.preprocessing import StandardScaler  #import du module 

scaler = StandardScaler() #instancie un objet StandardScaler
scaler.fit(X) #appliqué aux données
X_scaled = scaler.transform(X) #données transformées 

pcaDM = PCA(n_components=5) #instancie un objet PCA
pcaDM.fit(X_scaled)  #appliqué aux données scaled

pcaDM.components_.T
pcaDM.explained_variance_
pcaDM.explained_variance_ratio_   #en pourcentage
pcaDM.explained_variance_ratio_[0]
#Préparation des données pour affichage
dfpcaDM = pd.DataFrame(data = pcaDM.explained_variance_ratio_
             , columns = ['Variance Expliquée'])
dfpcaDM.index.name = 'Composantes'
dfpcaDM.reset_index(inplace=True)
dfpcaDM['Composantes'] +=1

#Graphique
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.barplot(x='Composantes', y= 'Variance Expliquée', data=dfpcaDM )
#fig.suptitle("La première composante comprend déja " + "{0:.2f}%".format(pca.explained_variance_ratio_[0]*100) + "de l'information", fontsize=10, fontweight='bold')
ax.set(xlabel='Composantes', ylabel='% Variance Expliquée',
       title="La première composante comprend " + "{0:.2f}%".format(pcaDM.explained_variance_ratio_[0]*100) + "de l'information")
fig.text(.9,-.05,"Screeplot du % de variance des composantes de l'ACP Normalisée \n Corrélation de Pearson pour les canaux Direct Marketing", fontsize=9, ha="right")
#plt.show()
fig.savefig("DM-PCA-StandardScaler-Pearson-screeplot-channel.png", bbox_inches="tight", dpi=600)


##############
##nuage des individus et axes des variables
#Labels
labels=DMPVDataForACP.columns.values  
score= X_scaled[:,0:2]
coeff=np.transpose(pcaDM.components_[0:2, :])
n = coeff.shape[0]

xs = score[:,0]
ys = score[:,1]  
#
scalex = 1.0/(xs.max() - xs.min())
scaley = 1.0/(ys.max() - ys.min())

#Graphique du nuage des pages et des axes des variables.
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot 
sns.scatterplot(xs * scalex,ys * scaley, alpha=0.4) #
for i in range(n):
    ax.arrow(0, 0, coeff[i,0]*1, coeff[i,1]*1,color = 'r',alpha = 0.5, head_width=.03)
    ax.text(coeff[i,0]*1.15, coeff[i,1]*1.15 , labels[i], color = 'r', ha = 'center', va = 'center')
ax.set(xlabel='Composante 1', ylabel='Composante 2',
       title="Les variables sont toutes du même bord de l'axe principal 1 \n Sur l'axe 2 Referral se détache légèrement")
ax.set_ylim((-0.7, 1.1))

fig.text(.9,-.05,"Nuage des pages et axes des variables Normalisées via StandardScaler \n Corrélation de Pearson pour les canaux Direct Marketing", fontsize=9, ha="right")
#plt.show()
fig.savefig("DM-PCA-StandardScaler-Pearson-cloud-channel.png", bbox_inches="tight", dpi=600)






##########################################################################
# MERCI pour votre attention !
##########################################################################
#on reste dans l'IDE
#if __name__ == '__main__':
#  main()

