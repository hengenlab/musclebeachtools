
def get_group_value(channel, group_size):
    '''
    get_group_value(channel, group_size)

    Calculates the group value of a channel
     by finding the remainder when divided by group_size.

    channel (int): The channel number.
    group_size (int): The number of groups.

    Group size must not be zero or negative
    channel must be non-negative

    Returns
        int: The group value (0 to group_size - 1).
'''

    # raise errors
    if channel < 0:
        raise ValueError("Channel must be non-negative")
    if group_size <= 0:
        raise ValueError("Group size must not be zero or negative")

    return int(channel % group_size)
