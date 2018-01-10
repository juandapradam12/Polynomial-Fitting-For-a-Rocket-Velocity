import numpy as np 
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

#### Datos

t=np.array([2.,5.,9.])
v=np.array([[45.948,119.985,231.497]])
vel=np.array([45.948,119.985,231.497])

#### Sistema de ecuaciones en forma matricial M_T*a=v

# Matriz del tiempo

M_t=np.array([[t[0]**2,t[0],1.],[t[1]**2,t[1],1.],[t[2]**2,t[2],1.]])

# El sistema de ecuaciones en forma matricial resulta en la matriz aumentada MA_t=v

MA_t=np.concatenate((M_t, v.T), axis=1)

#print(MA_t)

#### Solucion del Sistema de Ecuaciones con Gauss-Jordan (implementado)

def diagonaldeunos(A): # Deja a la matriz con unos en su diagonal
	n=A.shape[0]
	m=A.shape[1]
	M=np.ones((n,m), float)
	for i in range(0,n):
		if(A[i,i] != 0):
			M[i,:]=abs(A[i,:]/A[i,i])	
	return(M)


def gaussjordan(A):
	n=A.shape[0] 
	m=A.shape[1]
	M=np.ones((n,m), float)	
	M_F=np.ones((n,m), float)
	for i in range(0,n):
		for j in range(0,n-1): 
			if(i==0 and A[i,i] != 0): # deja la primera fila con 1 en su primera columna 
				M_F[i,:]=A[i,:]/A[i,i]
			elif(i>=1 and i>j): 
				if(j==0): # deja ceros en las filas mayores a la primera y columna cero 
					M[i,:]=A[i,:]-A[i,j]*M_F[j,:]
					M_F[i,:]=diagonaldeunos(M)[i,:]
				elif(j>0): # deja ceros en columnas mayores a la primera
					M[i,:]=M_F[i,:]-M_F[i,j]*M_F[j,:]
					M_F[i,:]=diagonaldeunos(M)[i,:]
	return(M_F)
 
#print(gaussjordan(MA_t))

# Valores de los coefs de acuerdo a la solucion del sistema de ecuaciones

a_3=gaussjordan(MA_t)[2,3]
a_2=gaussjordan(MA_t)[1,3]-gaussjordan(MA_t)[1,2]*a_3
a_1=gaussjordan(MA_t)[0,3]-gaussjordan(MA_t)[0,2]*a_3-gaussjordan(MA_t)[0,1]*a_2

#### Grafica 

def velocidad(t):
	return a_1*(t**2)+a_2*t+a_3

T=np.linspace(0,10,10)

plt.plot(t,vel,'ro')
plt.plot(T,velocidad(T))
plt.title('Ajuste de Polinomio')
plt.xlabel('tiempo (s)')
plt.ylabel('velocidad (m/s)')
plt.xlim(-1,11)
plt.ylim(-5,270)
plt.savefig('VelocidadCohete.pdf')
#plt.show()

#### Valores de los coefs y velocidad en t=7s 

## Imprime los valores de los coefs

print(a_1,a_2,a_3)

## Imprime v(7)

print(velocidad(7))



