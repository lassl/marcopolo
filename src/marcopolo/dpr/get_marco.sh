mkdir data
cd data
wget --header "X-Ms-Version: 2019-12-12" https://msmarco.blob.core.windows.net/msmarcoranking/msmarco_v2_doc.tar

tar -xvf msmarco_v2_doc.tar
rm -rf msmarco_v2_doc.tar
cd msmarco_v2_doc
gunzip **

cd ../
wget https://msmarco.blob.core.windows.net/msmarcoranking/docv2_train_queries.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/docv2_train_qrels.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/docv2_train_top100.txt.gz
gunzip docv2_train_top100.txt.gz