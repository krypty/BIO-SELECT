library(Biobase)
library(golubEsets)
library(limma)

data("Golub_Merge")
eset <- Golub_Merge
model <- model.matrix(~pData(eset)$ALL.AML)
fit <- eBayes(lmFit(log2(exprs(eset)), model), 0.01)
tT <- topTable(fit, adjust="fdr", sort.by="B", number=1000)
write.table(tT, file="/dataset/limma-golub.csv", sep="\t")
#write.fit(fit, file="/dataset/limma-golub.csv", adjust="fdr", method="global", sep="\t")

# fix header in csv file, add the ID column
system("sed -i -e 's/\"logFC\"/\"ID\"\t\"logFC\"/' /dataset/limma-golub.csv")
