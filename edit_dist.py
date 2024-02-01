import editdistance

def match_words_with_edit_distance(input, actual):
    # Calculate the Levenshtein distance between the lowercase versions of the words
    distance = editdistance.eval(input, actual)
    threshold = len(input) - len(actual)

    # Check if the distance is below the threshold
    if distance <= threshold:
        if threshold == distance:
          return True
        else:
          if input.find(actual) != -1:
            return True
          else :
            return False
    else:
        return False
#edit distance algorithm
def dice_coefficient(a,b):
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0

    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]

    a_bigram_list.sort()
    b_bigram_list.sort()

    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1

    score = float(matches)/float(lena + lenb)
    return score

fields=["documentStatusCode", 
"documentType", 
"publishFile", 
"isLatest", 
"k2Handle", 
"isActive", 
"title", 
"isCheckedOut", 
"isLive", 
"revision", 
"project", 
"documentStatusDescription", 
"disciplineDescription", 
"revisionFileName", 
"revisionDate", 
"plantCode", 
"latestChangeDateSubq", 
"discipline", 
"dataStore", 
"documentTypeDescription", 
"externalLink", 
"description"]
attributes=["discipline_code", 
"creation_date", 
"handover_required", 
"revision_date", 
"company_name", 
"cmpy_seq_nr", 
"document_description_240", 
"status_code", 
"title", 
"percentage_complete", 
"publish_file", 
"subclass_code", 
"maintained_by", 
"subclass_type_description", 
"sucl_seq_nr", 
"publication_file", 
"handover_completed", 
"RHLLEGDOC", 
"document_url", 
"document_nr", 
"TITLE", 
"subclass_description", 
"proj_seq_nr", 
"description", 
"is_latest", 
"latest_rev_creation_date", 
"email_attachment_list", 
"document_type_select", 
"actual_start_month", 
"clas_seq_nr", 
"asst_seq_nr", 
"publication_file_required", 
"asset_code", 
"RHLPROJECT", 
"revision_creation_date_month", 
"pk_seq_nr", 
"class_code", 
"document_remarks", 
"docs_seq_nr", 
"created_by", 
"DOCUMENT_NUMBER", 
"actual_start_date", 
"doty_seq_nr", 
"active_ind", 
"default_asst_seq_nr", 
"checked_out_ind", 
"project_code", 
"latest_revision_code", 
"revision_creation_date_week", 
"company_code", 
"dost_seq_nr", 
"latest_change_date_pub_file", 
"revision_code", 
"has_rev_outstanding_ce_copy", 
"actual_start_date_week", 
"last_refresh_date", 
"revision_order", 
"source_file_required", 
"suty_seq_nr", 
"discipline_description", 
"latest_rev_creation_date_print", 
"project_title", 
"originator", 
"source_file", 
"actual_start_date_print", 
"disc_seq_nr", 
"actual_start_week", 
"status_description", 
"reco_seq_nr", 
"class_description", 
"asset_description", 
"latest_dore_seq_nr", 
"maintenance_date", 
"company_document_nr", 
"free_text_ind", 
"subclass_type_code", 
"document_type", 
"document_type_description", 
"latest_change_date_subq", 
"latest_rev_creation_date_slt", 
"swp_parent_string", 
"dore_seq_nr", 
"in_maintenance" ]

def run_edit(fields,attributes):
    field_dict=dict()
    for field in fields:
        dice_values=[dice_coefficient(field,attribute) for attribute in attributes]
        edit_values=[match_words_with_edit_distance(field.lower(), attribute.lower()) for attribute in attributes]
        result=[(dice>=0.5 or edit) for dice,edit in zip(dice_values,edit_values)]
        field_dict[field]=[attributes[i] for i in range(len(attributes)) if result[i]]
    return field_dict
