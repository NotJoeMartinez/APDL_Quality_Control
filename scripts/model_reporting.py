import datetime as dt
""" Hacky way of getting quick statistics from model dataframe """
def calculate_results(df, class_names, model_path, report_name):
    now = dt.datetime.now().strftime("%m%d%-I%M%S")

    total_tests = df.shape[0]
    total_correct = df['score'].value_counts()['True'] 
    total_incorrect = df['score'].value_counts()['False']
    percent_correct = (float(total_correct) / float(total_tests)) * 100
    missed_labels = df[df['score']=='False'] #subset dataFrame

    print(f"DateTime Of Test: {now}")
    print(f"Model Path: {model_path}")
    print(f"Raw Test Data Path: notes/csvs/{report_name}.csv")
    print("=======================")
    print(f"Total Tests: {total_tests}")
    print(f"correct predictions: {total_correct}")
    print(f"incorrect predictions: {total_incorrect}")
    print(f"Percentage correct: {round(percent_correct, 2)}%")
    print("=======================")
    print("Most missed predictions")

    for class_name in class_names:
      try:
        print(f"{class_name}:  {missed_labels['actual'].value_counts()[class_name]}")
      except KeyError:
        pass




""""Makes a markdown summary in notes/{report_name.md}"""
def make_md_notes(model, df, report_name, class_names, model_path):
    import markdown
    from contextlib import redirect_stdout

    with open(f"notes/{report_name}.md", 'w') as f:
      f.write(f"## {report_name} \n\n")

      f.write(f"## Stats \n")
      with redirect_stdout(f):
        f.write("```\n")
        calculate_results(df, class_names, model_path, report_name)
        f.write("``` \n")
      f.write(f"### Confusion Matrix \n")
      f.write(f"![Confusion Matrix](imgs/{report_name}.png) \n")
      f.write(f"### Random Samples \n")

      f.write(f"![Random Samples](imgs/rand_samples_{report_name}.png) \n")

      f.write(f"### Model Summary \n")
      f.write("```")
      with redirect_stdout(f):
          model.summary()
      f.write("``` \n") 
      print(f"report saved in: notes/{report_name}.md")

