def sentence_retrieval(args, seeds, keywords):

    scores = defaultdict(dict)
    id2sent = {}
    id2start = {}
    id2end = {}
    with open(f'datasets/{args.dataset}/sentences.json') as fin:
        for idx, line in enumerate(fin):
            if idx % 10000 == 0:
                print(idx)
            data = json.loads(line)
            start = len(id2sent)
            end = start+len(data['sentences'])-1
            for sent in data['sentences']:
                sent_id = len(id2sent)
                id2sent[sent_id] = sent
                id2start[sent_id] = start
                id2end[sent_id] = end

                words = sent.split()
                word_cnt = defaultdict(int)
                for word in words:
                    word_cnt[word] += 1
                score = defaultdict(int)
                for seed in keywords:
                    for kw in keywords[seed]:
                        score[seed] += word_cnt[kw]
                pos_seeds = [x for x in score if score[x] > 0]
                if len(pos_seeds) == 1:
                    seed = pos_seeds[0]
                    scores[seed][sent_id] = score[seed]


    # print out top sentences
    topk = args.num_sent
    wd = args.sent_window
    top_sentences = []
    with open(f'datasets/{args.dataset}/top_sentences.json', 'w') as fout:
        for seed in seeds:
            out = {}
            out['seed'] = seed
            out['sentences'] = []
            scores_sorted = sorted(scores[seed].items(), key=lambda x: x[1], reverse=True)[:topk]
            # print(scores_sorted[-1])

            for k0, v in scores_sorted:
                out['sentences'].append(id2sent[k0])
                
                # for k in range(k0-1, k0-wd-1, -1):
                #     if k < id2start[k0]:
                #         break
                #     excl = 1
                #     for seed_other in seeds:
                #         if seed_other == seed:
                #             continue
                #         if k in scores[seed_other]:
                #             excl = 0
                #             break
                #     if excl == 1:
                #         out['sentences'].append(id2sent[k])
                #     else:
                #         break
                
                for k in range(k0-1, k0-wd-1, -1):
                    if k < id2start[k0]:
                        break
                    excl = 1
                    overlapping_seeds = []
                    for seed_other in seeds:
                        if seed_other == seed:
                            continue
                        if k in scores[seed_other]:
                            excl = 0
                            overlapping_seeds.append(seed_other)
                        if len(overlapping_seeds)>1:
                            break

                    if excl == 1:
                        out['sentences'].append(id2sent[k])
                    # condition of another topic(transition sentence)
                    elif excl == 0 and len(overlapping_seeds) == 1:
                        
                        seed_position=[]
                        other_seed_position=[]
                        # find word positions of keywords in keywords[seed], similar for other seed position in the sentence id2sent[k]
                        
                        for kw in keywords[seed]:
                            for word in id2sent[k].split():
                                if word == kw:
                                    seed_position.append(id2sent[k].index(word))

                        for kw in keywords[overlapping_seeds[0]]:
                            for word in id2sent[k].split():
                                if word == kw:
                                    other_seed_position.append(id2sent[k].index(word))
      
                                                
                        if min(other_seed_position) > max(seed_position):
                            # append the sentence from start to one position before "other_seed_position"(as in the greater than)
                            out['sentences'].append(id2sent[k][:other_seed_position])
                        elif max(other_seed_position) < min(seed_position):
                            # append the sentence from one position after other_seed_position to the end
                            out['sentences'].append(id2sent[k][other_seed_position + 1:])
                        else:
                            # append out sentences with the pattern that for example: seed_position is [1,3,7,8],other_seed_position is 5,then append sentence[0,(5-1)] and [(5+1),end]
                            out['sentences'].append(id2sent[k][:other_seed_position])
                            out['sentences'].append(id2sent[k][other_seed_position + 1:])
                        break
                                
                
                
                

                # for k in range(k0+1, k0+wd+1):
                #     if k > id2end[k0]:
                #         break
                #     excl = 1
                #     for seed_other in seeds:
                #         if seed_other == seed:
                #             continue
                #         if k in scores[seed_other]:
                #             excl = 0
                #             break
                #     if excl == 1:
                #         out['sentences'].append(id2sent[k])
                #     else:
                #         break
                
                for k in range(k0+1, k0+wd+1):
                    if k > id2end[k0]:
                        break
                    excl = 1
                    overlapping_seeds = []
                    for seed_other in seeds:
                        if seed_other == seed:
                            continue
                        if k in scores[seed_other]:
                            excl = 0
                            overlapping_seeds.append(seed_other)
                        if len(overlapping_seeds)>1:
                            break

                    if excl == 1:
                        out['sentences'].append(id2sent[k])
                    # condition of another topic(transition sentence)
                    elif excl == 0 and len(overlapping_seeds) == 1:
                        
                        seed_position=[]
                        other_seed_position=[]
                        # find word positions of keywords in keywords[seed], similar for other seed position in the sentence id2sent[k]
                        
                        for kw in keywords[seed]:
                            for word in id2sent[k].split():
                                if word == kw:
                                    seed_position.append(id2sent[k].index(word))

                        for kw in keywords[overlapping_seeds[0]]:
                            for word in id2sent[k].split():
                                if word == kw:
                                    other_seed_position.append(id2sent[k].index(word))
      
                                                
                        if min(other_seed_position) > max(seed_position):
                            # append the sentence from start to one position before "other_seed_position"(as in the greater than)
                            out['sentences'].append(id2sent[k][:other_seed_position])
                        elif max(other_seed_position) < min(seed_position):
                            # append the sentence from one position after other_seed_position to the end
                            out['sentences'].append(id2sent[k][other_seed_position + 1:])
                        else:
                            # append out sentences with the pattern that for example: seed_position is [1,3,7,8],other_seed_position is 5,then append sentence[0,(5-1)] and [(5+1),end]
                            out['sentences'].append(id2sent[k][:other_seed_position])
                            out['sentences'].append(id2sent[k][other_seed_position + 1:])
                        break
                
                
            fout.write(json.dumps(out)+'\n')
            top_sentences.append(out)
    return top_sentences
