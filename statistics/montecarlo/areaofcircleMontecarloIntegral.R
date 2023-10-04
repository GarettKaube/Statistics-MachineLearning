set.seed(250)
g <- function(x) {
  #distribution function 
  return((5/3)*(1-0.8*x))
}

#plotting the distribution function
x <- seq(0, 1, length = 1000)
plot(x, g(x))

n = 1000
generate_variables <- function(n) {
  #Acceptance-rejection method
  a <- 0 #acceptance
  y <- numeric(n) #values that have been accepted
  
  while (a < n) {
    u1 <- runif(1)
    u2 <- runif(1, 0, 5/3)
    if (g(u1) >= u2) {
      a  <- a + 1
      y[a] <- u1
    }
  }
  return(y)
}

random.n <- generate_variables(n)

n= 1000
set.seed(250)

qf <- function(x) {
  #quantile function for desired density
  return(-sqrt((25/16) - (3/2)*x) + 1.25)
}

inversion <- function(n) {
  #Generates random numbers for desired density
  u3 <- runif(n)
  rv <- qf(u3)
  return(rv)
}

inv <- inversion(1000)

#check the accept and reject method
qqplot(qf(seq(0,1, length = 1000)), random.n)
qqplot(qf(seq(0,1, length = 1000)), inv)



g <- function(x) {
  return(sqrt(1 - x^2))
}

n = 2000000
area_of_circleR1 <- function(n) {
  #Monte Carlo estimate of area of unit circle
  u1 <- runif(n,-1,1)
  u2 <- runif(n,0,1)
  y <- numeric(n)
  p <- mean(u2 <= sqrt(1 - u1^2))
  variance <- (p * (1-p) *4)/ n
  
  for (i in 1:n) {
    if (g(u1[i]) >= u2[i]) {
      y[i] <- 1
    }
  }
  
  sprintf("Area: %f,  Variance: %f", mean(y)*4, variance)
}

area_of_circleR1(2000000)
























