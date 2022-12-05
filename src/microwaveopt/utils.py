#  UTILITY FUNCTIONS
def find_line(lines, word):
    idx = []
    for l in range(len(lines)):
        if lines[l].strip().find(word) == 0:
            idx.append(l)
    return idx


def find_arg(line, word,  separator=' ', terminator=None):
    pt = line.lower().find(word.lower() + separator)
    if pt != -1:
        line = line[pt+len(word)+len(separator):]
        if terminator is not None:
            pt2 = line.find(terminator)
            if pt2 == -1:
                arg = line.strip()
            else:
                arg = line[:pt2]
        else:
            arg = line.strip()
    else:
        arg = None
    return arg

