def substring_ending(string):
    global max_value
    n = len(string)
    max_end=1
    for i in range(1,n):
        partial =substring_ending(string[:i])
        if string[i-1] < string[n-1] and partial + 1 > max_end:
            max_end = partial + 1
        if max_end > max_value:
            max_value = max_end
    return max_end

def longest_substring(string):
    global max_value
    max_value = 1
    substring_ending(string)
    return max_value


def longestSubstring(string):
    match=[]
    for i in range(len(string)):
        match.append(1)
    for i in range(0, len(string)):
        for j in range(0,i):
            if string[j] < string[i]:
                if match[i] < match[j] +1:
                    match[i] = match[j] +1
    max_value = max(match)
    return max_value
