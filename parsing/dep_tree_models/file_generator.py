from parsing.dep_tree_models.helpers.data_preparation import load_data_set

if __name__ == "__main__":

    for lang in ["lt_hse", "be_hse", "mr_ufal", "ta_ttb"]:
        for data_part in ["train", "test", "dev"]:
            for k in range(1, 16):
                load_data_set(lang=lang, data_part=data_part, k=k, tokenized=True)

        print(f"Data for {lang} prepared")