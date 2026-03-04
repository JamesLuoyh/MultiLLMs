import os
import pandas as pd
import numpy as np
import logging
import requests
import io

from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as hf_dataset

from typing import Iterable, Tuple, List, Union, Optional
from PIL import Image

log = logging.getLogger("lm_polygraph")


class Dataset:
    """
    Seq2seq or vision-language dataset for calculating quality of uncertainty estimation method.
    """

    def __init__(
        self, x: List[str], y: List[str], batch_size: int, images: Optional[str] = None
    ):
        """
        Parameters:
            x (List[str]): a list of input texts.
            y (List[str]): a list of output (target) texts. Must have the same length as `x`.
            batch_size (int): the size of the texts batch.
        """
        self.x = x
        self.y = y
        self.images = images
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[List[str], List[str], Optional[List]]]:
        """
        Returns:
            Iterable[Tuple[List[str], List[str]]]: iterates over batches in dataset,
                returns list of input texts and list of corresponding output texts.
        """
        for i in range(0, len(self.x), self.batch_size):
            batch_x = self.x[i : i + self.batch_size]
            batch_y = self.y[i : i + self.batch_size]
            batch_images = (
                self.images[i : i + self.batch_size]
                if self.images is not None
                else None
            )
            yield (batch_x, batch_y, batch_images)

    def __len__(self) -> int:
        """
        Returns:
            int: number of batches in the dataset.
        """
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def select(self, indices: List[int]):
        """
        Shrinks the dataset down to only texts with the specified index.

        Parameters:
            indices (List[int]): indices to left in the dataset.Must have the same length as input texts.
        """
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        if self.images is not None:
            self.images = [self.images[i] for i in indices]
        return self

    def train_test_split(self, test_size: int, seed: int, split: str = "train"):
        """
        Samples dataset into train and test parts.

        Parameters:
            test_size (int): size of test dataset,
            seed (int): seed to perform random splitting with,
            split (str): either 'train' or 'test'. If 'train', lefts only train data in the current dataset object.
                If 'test', left only test data. Default: 'train'.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: train input and target texts list,
                test input and target texts list.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(self.x),
            np.array(self.y),
            test_size=test_size,
            random_state=seed,
        )
        if self.images is not None:
            images_train, images_test = train_test_split(
                np.array(self.images), test_size=test_size, random_state=seed
            )
        else:
            images_train = images_test = None

        if split == "train":
            self.x, self.y, self.images = (
                X_train.tolist(),
                y_train.tolist(),
                images_train.tolist() if images_train is not None else None,
            )
        else:
            self.x, self.y, self.images = (
                X_test.tolist(),
                y_test.tolist(),
                images_test.tolist() if images_test is not None else None,
            )

        return (
            X_train.tolist(),
            X_test.tolist(),
            y_train.tolist(),
            y_test.tolist(),
        )

    def subsample(self, size: int, seed: int):
        """
        Subsamples the dataset to the provided size.

        Parameters:
            size (int): size of the resulting dataset,
            seed (int): seed to perform random subsampling with.
        """
        np.random.seed(seed)
        if len(self.x) < size:
            indices = list(range(len(self.x)))
        else:
            if size < 1:
                size = int(size * len(self.x))
            indices = np.random.choice(len(self.x), size, replace=False)
        self.select(indices)

    @staticmethod
    def from_csv(
        csv_path: str,
        x_column: str,
        y_column: str,
        batch_size: int,
        prompt: str = "",
        **kwargs,
    ):
        """
        Creates the dataset from .CSV table.

        Parameters:
            csv_path (str): path to .csv table,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch.
        """
        csv = pd.read_csv(csv_path)
        x = csv[x_column].tolist()
        y = csv[y_column].tolist()

        if len(prompt):
            x = [prompt.format(text=text) for text in x]

        return Dataset(x, y, batch_size)

    @staticmethod
    def load_hf_dataset(
        path: Union[str, List[str]],
        split: str,
        **kwargs,
    ):
        load_from_disk = kwargs.pop("load_from_disk", False)
        if load_from_disk:
            dataset_name = path
            dataset = hf_dataset.load_from_disk(path)
        elif isinstance(path, str):
            dataset_name = path
            dataset = load_dataset(path, split=split, **kwargs)
        else:
            dataset_name = path[0]
            dataset = load_dataset(*path, split=split, **kwargs)

        return dataset_name, dataset

    @staticmethod
    def from_datasets(
        dataset_path: Union[str, List[str]],
        x_column: str,
        y_column: str,
        batch_size: int,
        im_column: Optional[str] = None,
        prompt: str = "",
        description: str = "",
        mmlu_max_subject_size: int = 100,
        n_shot: int = 0,
        few_shot_split: str = "train",
        few_shot_prompt: Optional[str] = None,
        instruct: bool = False,
        split: str = "test",
        size: int = None,
        **kwargs,
    ):
        """
        Creates the dataset from Huggingface datasets.

        Parameters:
            dataset_path (str): HF path to dataset,
            x_column (str): name of column to take input texts from,
            y_column (str): name of column to take target texts from,
            batch_size (int): the size of the texts batch,
            prompt (str): prompt template to use for input texts (default: ''),
            split (str): dataset split to take data from (default: 'text'),
            size (Optional[int]): size to subsample dataset to. If None, the full dataset split will be taken.
                Default: None.
        """
        dataset_name, dataset = Dataset.load_hf_dataset(dataset_path, split, **kwargs)
        log.debug(f"Loaded HF dataset '{dataset_name}' split '{split}': {len(dataset)} samples")

        if size is not None:
            log.info(f"Size parameter provided: {size}, dataset length: {len(dataset)}")
            if size < len(dataset):
                log.info(f"Selecting first {size} samples from dataset")
                dataset = dataset.select(range(size))
                log.info(f"After selection: {len(dataset)} samples")
            else:
                log.info(f"Size {size} >= dataset length {len(dataset)}, using full dataset")
        else:
            log.info("No size parameter provided, using full dataset")

        if "allenai/c4" in dataset_name.lower():
            x, y = [], []
            for inst in dataset:
                if len(inst[x_column]) <= 1024:
                    x.append(inst[x_column])
                    y.append(inst[y_column])
        elif "medqa" in dataset_name.lower() or (isinstance(dataset_path, str) and "medqa" in dataset_path.lower()):
            # Special handling for MedQA-USMLE-4-options format
            # Format: question + options as input, answer_idx as target
            log.debug("Detected MedQA dataset format, formatting question and options")
            x, y = [], []
            for inst in dataset:
                # Format the question and options
                question = inst.get("question", "")
                options = inst.get("options", {})
                option_a = options.get("A", "")
                option_b = options.get("B", "")
                option_c = options.get("C", "")
                option_d = options.get("D", "")
                
                # Format the prompt
                if prompt:
                    formatted_input = prompt.format(
                        question=question.strip(),
                        option_a=option_a,
                        option_b=option_b,
                        option_c=option_c,
                        option_d=option_d,
                        text=question.strip(),  # For backward compatibility
                    )
                else:
                    # Default format if no prompt provided
                    formatted_input = (
                        f"Q: {question.strip()}\n"
                        f"A. {option_a}\n"
                        f"B. {option_b}\n"
                        f"C. {option_c}\n"
                        f"D. {option_d}\n"
                    )
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use answer_idx as target (A, B, C, or D)
                if y_column:
                    y.append(inst.get(y_column, inst.get("answer_idx", "")))
                else:
                    y.append(inst.get("answer_idx", ""))
            
            log.debug(f"Formatted {len(x)} MedQA samples")
        elif "gsm8k-mc" in dataset_name.lower() or (isinstance(dataset_path, str) and "gsm8k-mc" in dataset_path.lower()):
            # Special handling for GSM8K-MC format
            # Format: Question, A, B, C, D, Answer columns (separate columns, not dict)
            log.debug("Detected GSM8K-MC dataset format, formatting question and options")
            x, y = [], []
            for inst in dataset:
                # Get question and options from separate columns
                question = inst.get("Question", inst.get("question", ""))
                option_a = inst.get("A", "")
                option_b = inst.get("B", "")
                option_c = inst.get("C", "")
                option_d = inst.get("D", "")
                
                # Format the prompt
                if prompt:
                    formatted_input = prompt.format(
                        question=question.strip() if question else "",
                        option_a=option_a if option_a else "",
                        option_b=option_b if option_b else "",
                        option_c=option_c if option_c else "",
                        option_d=option_d if option_d else "",
                        text=question.strip() if question else "",  # For backward compatibility
                    )
                else:
                    # Default format if no prompt provided
                    formatted_input = (
                        f"Q: {question.strip() if question else ''}\n"
                        f"A. {option_a if option_a else ''}\n"
                        f"B. {option_b if option_b else ''}\n"
                        f"C. {option_c if option_c else ''}\n"
                        f"D. {option_d if option_d else ''}\n"
                    )
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use Answer column as target (A, B, C, or D)
                if y_column:
                    answer = inst.get(y_column, inst.get("Answer", inst.get("answer", "")))
                    y.append(answer)
                else:
                    y.append(inst.get("Answer", inst.get("answer", "")))
            
            log.debug(f"Formatted {len(x)} GSM8K-MC samples")
        elif "ai2_arc" in dataset_name.lower() or (isinstance(dataset_path, str) and "ai2_arc" in dataset_path.lower()) or (isinstance(dataset_path, list) and any("ai2_arc" in str(p).lower() for p in dataset_path)):
            # Special handling for ARC dataset format
            # Format: question, choices (dict with A, B, C, D, etc.), answerKey
            log.debug("Detected ARC dataset format, formatting question and options")
            x, y = [], []
            valid_answer_keys = {"A", "B", "C", "D"}
            skipped_count = 0
            
            for inst in dataset:
                # Get answer key and filter to only A, B, C, D
                answer_key = inst.get("answerKey", "")
                if answer_key not in valid_answer_keys:
                    skipped_count += 1
                    continue
                
                # Get question and choices
                question = inst.get("question", "")
                choices = inst.get("choices", {})
                
                # Extract options - ARC dataset format: choices is a dict with "label" and "text" lists
                # Format: {"label": ["A", "B", "C", "D"], "text": ["option1", "option2", "option3", "option4"]}
                option_a = ""
                option_b = ""
                option_c = ""
                option_d = ""
                
                if isinstance(choices, dict):
                    # Check if it's the ARC format with "label" and "text" lists
                    if "label" in choices and "text" in choices and isinstance(choices["label"], list):
                        # ARC format: {"label": ["A", "B", "C", "D"], "text": ["...", "...", "...", "..."]}
                        labels = choices.get("label", [])
                        texts = choices.get("text", [])
                        for label, text in zip(labels, texts):
                            if label == "A":
                                option_a = text
                            elif label == "B":
                                option_b = text
                            elif label == "C":
                                option_c = text
                            elif label == "D":
                                option_d = text
                    elif isinstance(choices, list):
                        # Format: [{"label": "A", "text": "..."}, {"label": "B", "text": "..."}, ...]
                        for choice in choices:
                            if isinstance(choice, dict):
                                label = choice.get("label", "")
                                text = choice.get("text", "")
                                if label == "A":
                                    option_a = text
                                elif label == "B":
                                    option_b = text
                                elif label == "C":
                                    option_c = text
                                elif label == "D":
                                    option_d = text
                    else:
                        # Format: {"A": "...", "B": "...", ...}
                        option_a = choices.get("A", "")
                        option_b = choices.get("B", "")
                        option_c = choices.get("C", "")
                        option_d = choices.get("D", "")
                
                # Format the prompt
                # Support both old format (option_a, option_b, etc.) and new bayesian-peft format (choices)
                if prompt:
                    # Check if prompt uses {choices} format (bayesian-peft style)
                    if "{choices}" in prompt:
                        # Format choices as "A) option_text\nB) option_text\n..." to match bayesian-peft
                        choices_list = []
                        for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
                            if text:  # Only include if option text is not empty
                                choices_list.append(f"{label}) {text}")
                        choices_str = "\n".join(choices_list)
                        formatted_input = prompt.format(
                            question=question.strip() if question else "",
                            choices=choices_str,
                            # Also support old format for backward compatibility
                            option_a=option_a if option_a else "",
                            option_b=option_b if option_b else "",
                            option_c=option_c if option_c else "",
                            option_d=option_d if option_d else "",
                            text=question.strip() if question else "",
                        )
                    else:
                        # Old format with individual options
                        formatted_input = prompt.format(
                            question=question.strip() if question else "",
                            option_a=option_a if option_a else "",
                            option_b=option_b if option_b else "",
                            option_c=option_c if option_c else "",
                            option_d=option_d if option_d else "",
                            text=question.strip() if question else "",  # For backward compatibility
                        )
                else:
                    # Default format: use bayesian-peft style as default for ARC
                    choices_list = []
                    for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
                        if text:
                            choices_list.append(f"{label}) {text}")
                    choices_str = "\n".join(choices_list)
                    formatted_input = (
                        f"Return the label of the correct answer for the question below.\n\n"
                        f"Question: {question.strip() if question else ''}\n"
                        f"Choices:\n{choices_str}\n"
                        f"Answer:"
                    )
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use answerKey as target (A, B, C, or D)
                if y_column:
                    y.append(inst.get(y_column, answer_key))
                else:
                    y.append(answer_key)
            
            log.debug(f"Formatted {len(x)} ARC samples (skipped {skipped_count} with answer keys other than A, B, C, D)")
        elif "medmcqa" in dataset_name.lower() or (isinstance(dataset_path, str) and "medmcqa" in dataset_path.lower()):
            # Special handling for MedMCQA dataset format
            # Format: question + opa/opb/opc/opd as options, cop (0-3) as correct answer
            log.debug("Detected MedMCQA dataset format, formatting question and options")
            x, y = [], []
            option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            skipped_count = 0
            
            # Log first example to debug structure
            if len(dataset) > 0:
                first_example = dataset[0]
                log.debug(f"MedMCQA dataset sample keys: {list(first_example.keys())}")
                log.debug(f"MedMCQA first example sample: {first_example}")
            
            # Count cop value distribution for debugging
            cop_distribution = {}
            for inst in dataset:
                cop = inst.get("cop", None)
                if cop is not None:
                    cop_distribution[cop] = cop_distribution.get(cop, 0) + 1
            if cop_distribution:
                log.info(f"MedMCQA cop value distribution: {cop_distribution}")
            
            for inst in dataset:
                # Format the question and options
                # Try multiple possible column names
                question = inst.get("question", inst.get("Question", ""))
                
                # Try different option column name variations
                option_a = inst.get("opa", inst.get("option_a", inst.get("A", "")))
                option_b = inst.get("opb", inst.get("option_b", inst.get("B", "")))
                option_c = inst.get("opc", inst.get("option_c", inst.get("C", "")))
                option_d = inst.get("opd", inst.get("option_d", inst.get("D", "")))
                
                # Get correct option (cop is 0, 1, 2, 3 for A, B, C, D)
                # Note: cop = -1 means "no answer" or "unknown" and should be skipped
                # Try multiple possible column names
                cop = inst.get("cop", inst.get("correct_idx", inst.get("correct_option", None)))
                if cop is not None:
                    # Convert 0-3 to A-D
                    # Handle both int and string representations
                    if isinstance(cop, str):
                        try:
                            cop = int(cop)
                        except (ValueError, TypeError):
                            cop = None
                    # Skip if cop is -1 (no answer) or not in valid range (0-3)
                    if cop == -1:
                        skipped_count += 1
                        continue
                    if cop is not None and cop in option_map:
                        answer_letter = option_map[cop]
                    else:
                        skipped_count += 1
                        continue
                else:
                    # Fallback to answer_idx if cop is not available
                    answer_idx = inst.get("answer_idx", inst.get("answer", ""))
                    if isinstance(answer_idx, int):
                        if answer_idx in option_map:
                            answer_letter = option_map[answer_idx]
                        else:
                            skipped_count += 1
                            continue
                    elif isinstance(answer_idx, str) and answer_idx.upper() in ["A", "B", "C", "D"]:
                        answer_letter = answer_idx.upper()
                    else:
                        skipped_count += 1
                        continue
                
                # Skip if we don't have valid question or options
                if not question or not (option_a or option_b or option_c or option_d):
                    skipped_count += 1
                    continue
                
                # Format the prompt
                # Support both old format (option_a, option_b, etc.) and new bayesian-peft format (choices)
                if prompt:
                    # Check if prompt uses {choices} format (bayesian-peft style)
                    if "{choices}" in prompt:
                        # Format choices as "A) option_text\nB) option_text\n..." to match bayesian-peft
                        choices_list = []
                        for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
                            if text:  # Only include if option text is not empty
                                choices_list.append(f"{label}) {text}")
                        choices_str = "\n".join(choices_list)
                        try:
                            formatted_input = prompt.format(
                                question=question.strip() if question else "",
                                choices=choices_str,
                                # Also support old format for backward compatibility
                                option_a=option_a if option_a else "",
                                option_b=option_b if option_b else "",
                                option_c=option_c if option_c else "",
                                option_d=option_d if option_d else "",
                                text=question.strip() if question else "",
                            )
                        except KeyError as e:
                            log.warning(f"Prompt format error: {e}, using default format")
                            formatted_input = (
                                f"Q: {question.strip() if question else ''}\n"
                                f"A. {option_a if option_a else ''}\n"
                                f"B. {option_b if option_b else ''}\n"
                                f"C. {option_c if option_c else ''}\n"
                                f"D. {option_d if option_d else ''}\n"
                            )
                    else:
                        # Old format with individual options
                        try:
                            formatted_input = prompt.format(
                                question=question.strip() if question else "",
                                option_a=option_a if option_a else "",
                                option_b=option_b if option_b else "",
                                option_c=option_c if option_c else "",
                                option_d=option_d if option_d else "",
                                text=question.strip() if question else "",  # For backward compatibility
                            )
                        except KeyError as e:
                            log.warning(f"Prompt format error: {e}, using default format")
                            formatted_input = (
                                f"Q: {question.strip() if question else ''}\n"
                                f"A. {option_a if option_a else ''}\n"
                                f"B. {option_b if option_b else ''}\n"
                                f"C. {option_c if option_c else ''}\n"
                                f"D. {option_d if option_d else ''}\n"
                            )
                else:
                    # Default format: use bayesian-peft style as default
                    choices_list = []
                    for label, text in [("A", option_a), ("B", option_b), ("C", option_c), ("D", option_d)]:
                        if text:
                            choices_list.append(f"{label}) {text}")
                    choices_str = "\n".join(choices_list)
                    formatted_input = (
                        f"Answer the multiple choice question below by returning the answer label (A to D)\n\n"
                        f"Question: {question.strip() if question else ''}\n"
                        f"Choices:\n{choices_str}\n"
                        f"Answer:"
                    )
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use answer_letter as target (A, B, C, or D)
                y.append(answer_letter)
            
            if len(x) == 0:
                # Check if all samples had cop=-1 (no answer)
                all_cop_neg_one = all(
                    inst.get("cop", None) == -1 
                    for inst in dataset
                )
                if all_cop_neg_one:
                    raise ValueError(
                        f"No valid MedMCQA samples found. All {skipped_count} samples have cop=-1 (no answer). "
                        f"The '{split}' split may not have labels. "
                        f"Try using '--split validation' or '--split train' instead, "
                        f"or check if the dataset split has labeled examples."
                    )
                else:
                    raise ValueError(
                        f"No valid MedMCQA samples found after processing. "
                        f"Skipped {skipped_count} samples. "
                        f"Please check the dataset format and column names. "
                        f"Expected cop values: 0, 1, 2, 3 (for A, B, C, D). "
                        f"cop=-1 indicates no answer and is skipped."
                    )
            log.debug(f"Formatted {len(x)} MedMCQA samples (skipped {skipped_count} invalid samples)")
        elif "cais/mmlu" in dataset_name.lower() or (isinstance(dataset_path, str) and "cais/mmlu" in dataset_path.lower()) or (isinstance(dataset_path, list) and any("cais/mmlu" in str(p).lower() for p in dataset_path)):
            # Special handling for cais/mmlu dataset format
            # Format: question + choices (A, B, C, D) as lists, answer as integer (0-3)
            log.debug("Detected cais/mmlu dataset format, formatting question and options")
            x, y = [], []
            option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            skipped_count = 0
            
            # Log first example to debug structure
            if len(dataset) > 0:
                first_example = dataset[0]
                log.debug(f"cais/mmlu dataset sample keys: {list(first_example.keys())}")
                log.debug(f"cais/mmlu first example sample: {first_example}")
            
            for inst in dataset:
                # Format the question and options
                question = inst.get("question", "")
                choices = inst.get("choices", [])
                answer = inst.get("answer", None)
                
                # Handle answer: can be integer (0-3) or already a letter
                if answer is None:
                    skipped_count += 1
                    continue
                
                if isinstance(answer, int):
                    if answer < 0 or answer >= len(choices):
                        skipped_count += 1
                        continue
                    answer_letter = option_map.get(answer, "")
                    if not answer_letter:
                        skipped_count += 1
                        continue
                elif isinstance(answer, str):
                    answer_letter = answer.upper()
                    if answer_letter not in {"A", "B", "C", "D"}:
                        skipped_count += 1
                        continue
                else:
                    skipped_count += 1
                    continue
                
                # Extract options - handle list of choices
                options = {}
                if isinstance(choices, list) and len(choices) >= 4:
                    options = {
                        "A": choices[0],
                        "B": choices[1],
                        "C": choices[2],
                        "D": choices[3],
                    }
                elif isinstance(choices, dict):
                    options = choices
                else:
                    skipped_count += 1
                    continue
                
                # Format the prompt
                # Support both old format (option_a, option_b, etc.) and new bayesian-peft format (choices)
                if prompt:
                    # Check if prompt uses {option_a} format or {choices} format
                    if "{choices}" in prompt:
                        # Format choices as "A) ... B) ... C) ... D) ..."
                        choices_str = "\n".join([f"{k}) {options.get(k, '')}" for k in ["A", "B", "C", "D"]])
                        formatted_input = prompt.format(
                            question=question,
                            choices=choices_str,
                            option_a=options.get("A", ""),
                            option_b=options.get("B", ""),
                            option_c=options.get("C", ""),
                            option_d=options.get("D", ""),
                        )
                    else:
                        # Try to format with individual options
                        formatted_input = prompt.format(
                            question=question,
                            option_a=options.get("A", ""),
                            option_b=options.get("B", ""),
                            option_c=options.get("C", ""),
                            option_d=options.get("D", ""),
                        )
                else:
                    # Default format if no prompt template provided
                    choices_str = "\n".join([f"{k}) {options.get(k, '')}" for k in ["A", "B", "C", "D"]])
                    formatted_input = f"Question: {question}\n{choices_str}"
                
                if description:
                    formatted_input = description + "\n\n" + formatted_input
                
                x.append(formatted_input)
                
                # Use answer_letter as target
                if y_column:
                    y.append(inst.get(y_column, answer_letter))
                else:
                    y.append(answer_letter)
            
            if len(x) == 0:
                raise ValueError(
                    f"No valid cais/mmlu samples found after processing. "
                    f"Skipped {skipped_count} samples. "
                    f"Please check the dataset format and column names. "
                    f"Expected columns: 'question', 'choices' (list), 'answer' (int 0-3). "
                    f"Split: {split}"
                )
            
            log.debug(f"Formatted {len(x)} cais/mmlu samples (skipped {skipped_count} invalid samples)")
        else:
            # Convert to lists to ensure we only have the selected samples
            # This is important because dataset[x_column] on a lazy dataset might access all rows
            log.debug(f"Converting dataset to lists (current length: {len(dataset)})")
            x = list(dataset[x_column])
            if y_column is not None:
                y = list(dataset[y_column])
            else:
                y = ["" for _ in range(len(x))]
            log.info(f"Extracted {len(x)} samples from dataset")

        images = dataset[im_column] if im_column else None
        if images is not None:
            images = list(images)

        return Dataset(x, y, batch_size, images=images)

    @staticmethod
    def load(path_or_path_and_files: Union[str, List[str]], *args, **kwargs):
        """
        Creates the dataset from either local .csv path (if such exists) or Huggingface datasets.
        See `from_csv` and `from_datasets` static functions for the description of *args and **kwargs arguments.

        Parameters:
            path_or_path_and_files (str or List[str]): local path to .csv table or HF path to dataset.
        """
        if isinstance(path_or_path_and_files, str) and os.path.isfile(
            path_or_path_and_files
        ):
            return Dataset.from_csv(path_or_path_and_files, *args, **kwargs)
        return Dataset.from_datasets(path_or_path_and_files, *args, **kwargs)

    @staticmethod
    def get_images(images: List[Union[Image.Image, str, bytes]]):
        imgs: List[Image.Image] = []
        for image_input in images:
            try:
                if isinstance(image_input, Image.Image):
                    imgs.append(image_input.convert("RGB"))
                elif isinstance(image_input, str) and image_input.startswith("http"):
                    response = requests.get(image_input, stream=True, timeout=10)
                    response.raise_for_status()
                    imgs.append(Image.open(io.BytesIO(response.content)).convert("RGB"))
                elif isinstance(image_input, str):
                    imgs.append(Image.open(image_input).convert("RGB"))
                elif isinstance(image_input, (bytes, bytearray)):
                    imgs.append(Image.open(io.BytesIO(image_input)).convert("RGB"))
                else:
                    log.warning(f"Unsupported image input format: {type(image_input)}")
            except Exception as e:
                log.warning(f"Failed to load image '{image_input}': {e}")
        return imgs
