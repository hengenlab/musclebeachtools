
def get_group_value(channel, group_size):
    '''
    get_group_value(channel, group_size)

    Calculates the group value of a channel
     by finding the remainder when divided by group_size.

    channel (int): The channel number.
    group_size (int): The number of groups.

    group_size must be greater than 0.

    Returns
        int: The group value (0 to group_size - 1).
'''


    return int(channel % group_size)
