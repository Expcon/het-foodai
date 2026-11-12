"""Explicit R2-only tail merging: one shuffled pass, no dropped/duplicated samples."""
class MergeShortTailBatchSampler:
    def __init__(self,sampler,batch_size=32,min_tail_size=8):
        if type(batch_size) is not int or type(min_tail_size) is not int or not 1<=min_tail_size<=batch_size:
            raise ValueError('Invalid tail policy sizes')
        if len(sampler)<min_tail_size:raise ValueError('Insufficient samples for a safe tail batch')
        self.sampler=sampler;self.batch_size=batch_size;self.min_tail_size=min_tail_size

    def __len__(self):
        full,tail=divmod(len(self.sampler),self.batch_size)
        return full+int(tail>=self.min_tail_size) if full else 1

    def __iter__(self):
        pending=[];batch=[]
        for index in self.sampler:
            batch.append(index)
            if len(batch)==self.batch_size:
                if pending:yield pending
                pending=batch;batch=[]
        if batch and pending and len(batch)<self.min_tail_size:
            yield pending+batch
        else:
            if pending:yield pending
            if batch:yield batch
