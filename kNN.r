kNN <- function(inVec,trainSet,trainLabel,k){
        pair_dist <- apply(trainSet,1,function(x) sqrt(sum((inVec - x) ^ 2)))
        bind_dist <- cbind(pair_dist,trainLabel)
        pred <- names(sort(table(bind_dist[order(bind_dist[,1])[1:k],2]),decreasing=TRUE)[1])
}
