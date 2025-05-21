Hi there, I'm Ose


project:
  type: book
  output-dir: _site

book:
  title: "Python Git Course"
  author: "Your Name"
  date: "2025-05-21"
  chapters:
    - index.qmd

    - part: "Week 0"
      chapters:
        - python_course/week_0/software_list.md

    - part: "Week 1"
      chapters:
        - python_course/week_1/week_1_newsletter.md

    - part: "Week 2"
      chapters:
        - python_course/week_2/week_2_newsletter.md

    - part: "Week 3"
      chapters:
        - python_course/week_3/week_3_newsletter.md

    - part: "Week 4"
      chapters:
        - python_course/week_4/week_4_newsletter.md

    - part: "Week 5"
      chapters:
        - python_course/week_5/data_cleaning_demo.ipynb

    - part: "Week 6"
      chapters:
        - python_course/week_6/control_flow.ipynb
        - python_course/week_6/loops.ipynb

    - part: "Week 7"
      chapters:
        - python_course/week_7/functions_demo.ipynb
        - python_course/week_7/v1_importing_custom_functions.ipynb
        - python_course/week_7/v2_importing_custom_functions.ipynb

    - part: "Week 8"
      chapters:
        - python_course/week_8/grading_example.ipynb
        - python_course/week_8/polio_project.ipynb

format:
  html:
    theme: cosmo
    toc: true
    number-sections: true

editor: visual

publish:
  method: gh-pages
