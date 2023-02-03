def get_ariaid(file_name: str, return_char_start=False) -> str:
    aria_id_end_idx = file_name.rfind("]")
    aria_id_start_idx = file_name.rfind("[", 0, aria_id_end_idx - 1) + 1
    aria_id = file_name[aria_id_start_idx:aria_id_end_idx]
    if return_char_start:
        return aria_id, aria_id_start_idx
    return aria_id
