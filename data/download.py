from modelscope.msdatasets import MsDataset

ds =  MsDataset.load('DatatangBeijing/90000sets-Multi-domainCustomerServiceDialogueTextData', 
                    subset_name='default', 
                    split='train', 
                    cache_dir='./raw')