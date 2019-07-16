#python data_pro.py -d electronics
#python data_pro.py -d tv
#python data_pro.py -d kindle
#python data_pro.py -d toys
#python data_pro.py -d music
#python data_pro.py -d cds
#python data_pro.py -d yelp

python data_pro_for_nnsum.py -d music -o onmt
python data_pro_for_nnsum.py -d music -o nnsum
python data_pro_for_nnsum.py -d cds -o onmt
python data_pro_for_nnsum.py -d cds -o nnsum
python data_pro_for_nnsum.py -d toys -o onmt
python data_pro_for_nnsum.py -d toys -o nnsum
python data_pro_for_nnsum.py -d tv -o onmt
python data_pro_for_nnsum.py -d tv -o nnsum
python data_pro_for_nnsum.py -d kindle -o onmt 
python data_pro_for_nnsum.py -d kindle -o nnsum 

python data_pro_for_nnsum.py -d yelp -o nnsum
python data_pro_for_nnsum.py -d electronics -o nnsum

python data_pro_for_nnsum.py -d yelp -o onmt
python data_pro_for_nnsum.py -d electronics -o onmt
