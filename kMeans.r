tryKmeans <- function(infile,k,itermax=1000){
  data <- read.table(infile)
  result <- kmeans(as.matrix(data),iter.max=itermax,k)
  result.cluster <- result$cluster
}
