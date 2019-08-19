def is_in_component(scc, graph_embedding, largest_scc):
    # is scc an IN component to largest_scc?
    for src in scc:
        for tar in graph_embedding[src]:
            if tar in largest_scc:
                return True
    return False


def is_out_component(scc, graph_embedding, largest_scc):
    # is scc an OUT component to largest_scc?
    for src in largest_scc:
        for tar in graph_embedding[src]:
            if tar in scc:
                return True
    return False
