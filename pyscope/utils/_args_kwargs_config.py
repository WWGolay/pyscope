def _kwargs_to_config(kwargs):
    if kwargs is None or len(kwargs) == 0:  # pragma: no cover
        return ""
    string = ""
    for key, value in kwargs.items():
        if ":" in str(value):
            string += str(key) + "='" + str(value) + "',"
        else:
            string += str(key) + "=" + str(value) + ","
    return string[:-1]
