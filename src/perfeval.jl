# Performance evaluation

## correctrate & errorrate

correctrate(gt::IntegerVector, r::IntegerVector) = counteq(gt, r) / length(gt)
errorrate(gt::IntegerVector, r::IntegerVector) = countne(gt, r) / length(gt)

## 