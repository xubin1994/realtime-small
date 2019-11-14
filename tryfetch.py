import nori2 as nori
nross = nori.Fetcher()
import pickle
data_id = '3724481,1000ffa8dd85'
r = pickle.loads(nross.get(data_id))
print(r)
