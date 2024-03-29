import json
import argparse
import matplotlib.pyplot as plt

def calculate_stats(input_file):
    first_word_count, last_word_count, whisper_wer_count, max_ngrams_not_in_pred_count,max_ngrams_not_in_target_count, president_count   = 0, 0, 0, 0 , 0, 0
    num_words_target_count, zero_count = 0, 0
    whisper_wer_values = []
    
    total_count = 0
    with open(input_file, 'r') as f:
        for line in f:
            total_count += 1
            data = json.loads(line.strip())
            
            first_word_count += int(data['first_word_predicted'] == 0)
            last_word_count += int(data['last_word_predicted'] == 0)
            
            whisper_wer = data['whisper_wer']
            whisper_wer_values.append(whisper_wer)
            if whisper_wer > 0.5:
                whisper_wer_count += 1
            
            num_words_target = data['num_words_target']
            min_words_predicted = data['min_words_predicted']
            max_words_predicted = data['max_words_predicted']
            max_ngrams_not_in_pred = data['max_ngrams_not_in_pred']
            max_ngrams_not_in_target = data['max_ngrams_not_in_target']

            ngram_max = 3
            if max_ngrams_not_in_pred > ngram_max:
                max_ngrams_not_in_pred_count += 1
          
            if max_ngrams_not_in_target > ngram_max:
                max_ngrams_not_in_target_count += 1

            if data['president'] == 1:
                president_count +=1
            
            buffer = max(int(num_words_target/2),5)
            if not (min_words_predicted - buffer <= num_words_target <= max_words_predicted + buffer):
                num_words_target_count += 1
            
            if any([
                data['first_word_predicted'] == 0, 
                data['last_word_predicted'] == 0, 
                data['max_ngrams_not_in_pred'] > 3,
                data['max_ngrams_not_in_target'] > 3,
                data['president'] == 1
                
                
                #data['whisper_wer'] > 0.8, 
                #data['num_words_target']  < data['min_words_predicted'] - buffer or data['num_words_target'] > data['max_words_predicted'] + buffer
            ]):
                zero_count += 1
                
    # Calculate percentages
    first_word_percent = (first_word_count / total_count) * 100
    max_ngrams_not_in_pred_percent = (max_ngrams_not_in_pred_count / total_count) * 100
    max_ngrams_not_in_target_percent = (max_ngrams_not_in_target_count / total_count) * 100
    last_word_percent = (last_word_count / total_count) * 100
    president_percent = (president_count / total_count) * 100
    whisper_wer_percent = (whisper_wer_count / total_count) * 100
    num_words_target_percent = (num_words_target_count / total_count) * 100
    zero_percent = (zero_count / total_count) * 100
    
    # Print percentages
    print(f"Percentage of first_word_predicted being 0: {first_word_percent:.1f}%")
    print(f"Percentage of last_word_predicted being 0: {last_word_percent:.1f}%")
    #print(f"Percentage of whisper_wer above 0.8: {whisper_wer_percent:.1f}%")
    print(f"Percentage of max_ngrams_not_in_pred being above {ngram_max}: {max_ngrams_not_in_pred_percent:.1f}%")
    print(f"Percentage of max_ngrams_not_in_target being above {ngram_max}: {max_ngrams_not_in_target_percent:.1f}%")
    print(f"Percentage of president being 1: {president_percent:.1f}%")
    #print(f"Percentage num_words_target is outside min_words_predicted and max_words_predicted: {num_words_target_percent:.1f}%")
    print(f"Percentage where at least one of the specified fields is 0: {zero_percent:.1f}%")
    
    # Plot histogram for whisper_wer
    plt.hist(whisper_wer_values, bins=10, alpha=0.5, color='g')
    plt.title("Histogram of whisper_wer")
    plt.xlabel("whisper_wer")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate descriptive statistics from JSON lines file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON lines file.")
    args = parser.parse_args()
    
    calculate_stats(args.input_file)
