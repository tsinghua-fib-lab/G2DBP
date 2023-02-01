def min_group(orders, max_parts):
    l, map_ = zip(*sorted([(len(j), i) for i, j in enumerate(orders)], key=lambda x: (-x[0], x[1])))
    assert all(i <= max_parts for i in l)
    bins = []
    for id_, i in enumerate(l):
        for b in bins:
            if b[1] >= i:
                b[0].append(id_)
                b[1] -= i
                break
        else:
            bins.append([[id_], max_parts - i])
    return [[map_[j] for j in i[0]] for i in bins]
