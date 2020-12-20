import re
from collections import defaultdict

class ParseError(Exception):
    def __init__(self, i,j, msg=None):
        if msg is not None:
            self.message=msg+f" at ({i},{j})\n"
        super().__init__(self.message)



class XMLParser():


    def __init__(self, tokenizer:object = None):
        self.tokenizer = tokenizer
        self.tag_pattern = re.compile("<(/*?[-a-zA-Z]+)>")
        self.BEGIN_TAG = 0
        self.END_TAG = 1


    def parse(self, content):
        all_tag_matches = self.tag_pattern.finditer(content)

        # Validity array which stores whether the XML markers found in the document are valid or not
        is_valid = []

        # [(tag_name, position_index), ...] => Temporary storage for begin tags, which are popped when an associated end tag is found
        buffer = []

        # [(tag_name, tag_type, tag_start_idx, tag_end_idx)] => XML markers and its metadata
        markers = []

        # Mark validity of all XML markers found in the document
        for i,match in enumerate(all_tag_matches): 
            tag_name = match.groups()[0].lstrip("/")
            tag_type = self.END_TAG if match.groups()[0].startswith("/") else self.BEGIN_TAG

            markers.append((tag_name, tag_type, match.start(), match.end()))

            # Every marker is considered invalid initially
            is_valid += [0]

            if tag_type==self.END_TAG:
                # Start scanning the buffer for the begin tag from the end
                k=-1 
                
                try:
                    while buffer[k][0]!=tag_name:
                        k-=1
                except IndexError:

                    # No begin tag is found in the buffer for the current end tag
                    raise ParseError(match.start(), match.end(), msg=f"<{tag_name}> not found for </{tag_name}>")

                # Mark both the begin and tags as valid
                begin_tag_idx = buffer[k][1]
                is_valid[begin_tag_idx]=1
                is_valid[i]=1
                
                # Remove begin tag from the buffer
                del buffer[k]

            else:
                buffer.append((tag_name, i))
        
        markers = [marker for validity, marker in zip(is_valid, markers) if validity==1]

        # annotations => {"entity1":[ [start_token_idx, end_token_idx], ...], ...}
        annotations = defaultdict(list)

        # doc_tokens => list of tokens in the entire document
        doc_tokens = []
        total_num_tokens = 0

        # Tracks start token index for each entity type
        start_pos = 0
        
        for k in range(len(markers)-1):
            tag = markers[k][0]
            tag_type = markers[k][1]

            i = markers[k][3]
            j = markers[k+1][2]

            text_betn_consequent_tags = content[i:j]
            tokens = self.tokenizer(text_betn_consequent_tags)
            num_tokens = len(tokens)

            doc_tokens.extend(tokens)

            if tag_type==self.BEGIN_TAG:
                annotations[tag].append([start_pos])
            elif tag_type==self.END_TAG:
                annotations[tag][-1].append(total_num_tokens-1)

            start_pos += num_tokens
            total_num_tokens += num_tokens

        last_tag = markers[-1][0]
        annotations[last_tag][-1].append(total_num_tokens-1)

        return doc_tokens, annotations
