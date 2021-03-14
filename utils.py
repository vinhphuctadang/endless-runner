def textcolor_display(text, type_mes='er'):
    '''
    show text in format with color
    Agr:
        text: text to format with color
        type_mes : type of messgage could be 'er' or 'inf'
    return
        text formatted with color
    '''
    import platform
    end = '\033[00m'
    if type_mes in ['er', 'error']:

        if platform.system() == 'Windows':
            begin = ''
            end = ''
        else:
            begin = '\033[91m'
        return begin + text + end

    if type_mes in ['inf', 'information']:
        if platform.system() == 'Windows':
            begin = ''
            end = ''
        else:
            begin = '\033[96m'
        return begin + text + end

    if type_mes in ['normal']:
        return text
