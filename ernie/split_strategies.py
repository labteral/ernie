#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re


class RegexExpressions:
    split_by_dot = re.compile(r'[^.]+(?:\.\s*)?')
    split_by_semicolon = re.compile(r'[^;]+(?:\;\s*)?')
    split_by_colon = re.compile(r'[^:]+(?:\:\s*)?')
    split_by_comma = re.compile(r'[^,]+(?:\,\s*)?')

    url = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    domain = re.compile(r'\w+\.\w+')


class SplitStrategy:
    def __init__(self, split_patterns, remove_patterns=None, group_splits=True, remove_too_short_groups=True):
        if not isinstance(split_patterns, list):
            self.split_patterns = [split_patterns]
        else:
            self.split_patterns = split_patterns

        if remove_patterns is not None and not isinstance(remove_patterns, list):
            self.remove_patterns = [remove_patterns]
        else:
            self.remove_patterns = remove_patterns

        self.group_splits = group_splits
        self.remove_too_short_groups = remove_too_short_groups

    def split(self, text, tokenizer, split_patterns=None):
        if split_patterns is None:
            if self.split_patterns is None:
                return [text]
            split_patterns = self.split_patterns

        def len_in_tokens(text_):
            no_tokens = len(tokenizer.encode(text_, add_special_tokens=False))
            return no_tokens

        no_special_tokens = len(tokenizer.encode('', add_special_tokens=True))
        max_tokens = tokenizer.max_len - no_special_tokens

        if self.remove_patterns is not None:
            for remove_pattern in self.remove_patterns:
                text = re.sub(remove_pattern, '', text).strip()

        if len_in_tokens(text) <= max_tokens:
            return [text]

        selected_splits = []
        splits = map(lambda x: x.strip(), re.findall(split_patterns[0], text))

        aggregated_splits = ''
        for split in splits:
            if len_in_tokens(split) > max_tokens:
                if len(split_patterns) > 1:
                    sub_splits = self.split(split, tokenizer, split_patterns[1:])
                    selected_splits.extend(sub_splits)

                else:
                    selected_splits.append(split)

            else:
                if not self.group_splits:
                    selected_splits.append(split)

                else:
                    new_aggregated_splits = f'{aggregated_splits} {split}'.strip()
                    if len_in_tokens(new_aggregated_splits) <= max_tokens:
                        aggregated_splits = new_aggregated_splits

                    else:
                        selected_splits.append(aggregated_splits)
                        aggregated_splits = split

        if aggregated_splits:
            selected_splits.append(aggregated_splits)

        remove_too_short_groups = len(selected_splits) > 1 \
                                  and self.group_splits \
                                  and self.remove_too_short_groups

        if not remove_too_short_groups:
            final_splits = selected_splits
        else:
            final_splits = []
            min_length = tokenizer.max_len / 2
            for split in selected_splits:
                if len_in_tokens(split) >= min_length:
                    final_splits.append(split)

        return final_splits


class SplitStrategies:
    SentencesWithoutUrls = SplitStrategy(split_patterns=[
        RegexExpressions.split_by_dot, RegexExpressions.split_by_semicolon, RegexExpressions.split_by_colon,
        RegexExpressions.split_by_comma
    ],
                                         remove_patterns=[RegexExpressions.url, RegexExpressions.domain],
                                         remove_too_short_groups=False,
                                         group_splits=False)

    GroupedSentencesWithoutUrls = SplitStrategy(split_patterns=[
        RegexExpressions.split_by_dot, RegexExpressions.split_by_semicolon, RegexExpressions.split_by_colon,
        RegexExpressions.split_by_comma
    ],
                                                remove_patterns=[RegexExpressions.url, RegexExpressions.domain],
                                                remove_too_short_groups=True,
                                                group_splits=True)
