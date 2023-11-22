# to get list of dictionaries
def collate_fn(batch):
    return tuple(zip(*batch))