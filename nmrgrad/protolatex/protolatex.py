#!/usr/bin/env python3
import sys
import uuid
from lxml import etree

class xmlPrettyTable2col:
    class_MT_sum = 0.0
    classIx = 1
    stepIx = 1
    classtype = r"Class "
    steptype = r"Step "
    numb_cell = r"2.0cm"
    unit_cell = r"2.0cm"
    __Wcell = r"\cellcolor{white}"
    subclassIx = None

    def __init__(self, filename=None, classIx=0, subclassIx=None):
        self.classIx = classIx
        self.subclassIx = subclassIx
        self.filename = filename

    def printtex(self):
        self.tree = etree.parse(self.filename).getroot()
        self._xmlPrettyTable2col(self.tree)
        self._reset_counters()

    def _add_etree_uuid(self, xmlETree):
        for xmlETreeChild in xmlETree.getchildren():
            if xmlETreeChild.tag == "class":
                if "ref" not in xmlETreeChild.keys():
                    xmlETreeChild.attrib["ref"] = str(
                        uuid.uuid4().fields[-1])[:8]
                self._add_etree_uuid(xmlETreeChild)
            if xmlETreeChild.tag == "step":
                if "ref" not in xmlETreeChild.keys():
                    xmlETreeChild.attrib["ref"] = str(
                        uuid.uuid4().fields[-1])[:8]
        print(r"\vspace*{10mm}\pagebreak[0]")

    def _reset_counters(self):
        print(r"\setcounter{rclassno}{1}")
        print(r"\setcounter{rstepno}{1}")

    def _xmlPrettyTable2col(self, xmlETree):
        """Parse the Latex table.

        DESCRIPTION
        """
        for xmlETreeChild in xmlETree.getchildren():
            if xmlETreeChild.tag == "class":
                self._xmlPrettyProcessClass(xmlETreeChild)
                self._xmlPrettyTable2col(xmlETreeChild)
            if xmlETreeChild.tag == "step":
                self._xmlPrettyProcessStep(xmlETreeChild)

    def _xmlPrettyProcessClass(self, xmlETreeChild):
        # total machine time of process class:
        for xmlETreeChildClass in xmlETreeChild.getchildren():
            if "mt" in xmlETreeChildClass.keys():
                self.class_MT_sum += float(xmlETreeChildClass.attrib["mt"])

        # allow subclass
        print(r"\setcounter{rclassno}{" + str(self.classIx - 1) + r"}")
        classname = self.classtype + str(self.classIx)
        if self.subclassIx is None:
            classname = classname + "." + str(self.subclassIx)

        if "ref" in xmlETreeChild.keys():
            ref_class_name = str(xmlETreeChild.attrib["ref"])
        else:
            ref_class_name = str(
                "test" + str(self.classIx) + str(uuid.uuid1())[0:5])
        print(r" \rclass{" + ref_class_name + r"}")

        self.table_header()
        print(r"\toprule")
        print(r"{\bfseries\normalsize " + classname + r": } & "
              + r"\multicolumn{4}{X}{\bfseries\normalsize "
              + str(xmlETreeChild.attrib["name"]) + r"}")
        print(r"& \multicolumn{1}{>{\raggedright}r}{"
              + r"\hfill $\Sigma$ Machine Time (MT): \SI{"
              + str(int(self.class_MT_sum))
              + r"}{\minute}} \\")
        print(r"\midrule\end{tabularx}\nopagebreak" + "\r\n%\r\n")

    def _xmlPrettyProcessStep(self, xmlETreeChild):
        # MACHINE TIME
        print_mt = r" "
        if "mt" in xmlETreeChild.keys():
            self.class_MT_sum += float(xmlETreeChild.attrib["mt"])
            print_mt = xmlETreeChild.attrib["mt"]

        if "ref" in xmlETreeChild.keys():
            ref_counter_name = str(xmlETreeChild.attrib["ref"])
        else:
            ref_counter_name = str(
                "test" + str(self.classIx) + r"." + str(self.stepIx))
        print(r"\rstep{" + ref_counter_name + r"}")

        stepname = str(self.classIx) + r"." + str(self.stepIx)
        if self.subclassIx is not None:
            print(r"\setcounter{rsubstepno}{" +
                  str(self.subclassIx - 1) + r"}\nopagebreak")
            if "subref" in xmlETreeChild.keys():
                subref_counter_name = str(xmlETreeChild.attrib["subref"])
            else:
                subref_counter_name = r"subref_" + str(ref_counter_name)
            stepname = str(self.classIx) + r"." + \
                str(self.subclassIx) + r"." + str(self.stepIx)
            print(r"\rsubstep{" + subref_counter_name + r"}\nopagebreak")

        # Generate Parameter List:
        parameter_list = list()
        parameter_list_sort = list()
        for parameter in xmlETreeChild.getchildren():
            if parameter.tag == "parameter":
                parameter_list.append(parameter)
                parameter_list_sort.append(parameter.attrib["text"])

        # get sorted index and len
        parameter_list_sorted_idx = sorted(
            range(len(parameter_list_sort)),
            key=lambda k: parameter_list_sort[k])
        table_row_len = len(parameter_list_sorted_idx)

        print(r"\rowcolors{2}{gray!8}{gray!20}\nopagebreak")

        self.table_header()
        print(r"{\bfseries\small " + self.steptype
              + stepname + r": } & {\bfseries\small "
              + xmlETreeChild.attrib["name"] + r"} &  ")

        step_device = None
        for children in xmlETreeChild.getchildren():
            if children.tag == "device":
                step_device = children.text
            break
        if step_device:
            print(r"\multicolumn{3}{p{\dimexpr "
                  + self.numb_cell + r" + "
                  + self.unit_cell + r" + "
                  + self.numb_cell + r" + 4\tabcolsep}}{"
                  + str(step_device)
                  + r"\hfill} & ")  # \dimexpr\textwidth -1.2pt -
        else:
            print(r" &&& ")
        print(r"\multicolumn{1}{>{\raggedright}p{" + self.unit_cell
              + r"}}{MT: \hfill \SI{" + str(print_mt) + r"}{\minute} }")
        print(r"\\")

        # create the rows for process step
        for row_idx in range(table_row_len):
            print(self._parameterchildrow2col(
                parameter_list[parameter_list_sorted_idx[row_idx]]))

        for comment in xmlETreeChild.getchildren():
            if comment.tag == "comment":
                ctext = str(comment.text)
                # ctext = r"\begin{minipage}{\dimexpr \textwidth - 4\tabcolsep
                # - " + self.unit_cell + "}\n" + self.__Wcell
                # + comment.text + r"\end{minipage}"
                # print (self.__Wcell + r" &  \multicolumn{5}{l}{ " \
                # + ctext + r"} \\")
                p_1col = r"p{\dimexpr\textwidth - " + \
                    self.unit_cell + r" - 4\tabcolsep}"
                print(self.__Wcell + r" & "
                      + r"\multicolumn{5}{" + p_1col + "}{ "
                      + self.__Wcell + ctext + r"} \\")
        print(r"\end{tabularx}" + "\r\n%\r\n" + r"\\[.2cm]")
        self.stepIx += 1

    def _parameterchildrow2col(self, parameter_child):
        return_value = " "
        if "unit" in parameter_child.keys():  # \multicolumn{1}{l}{
            return_value = r"{" + parameter_child.attrib["text"] + ":"
            if "unit1" in parameter_child.keys():
                return_value += r"} &" \
                    + parameter_child.attrib["value1"] + r" & " \
                    + parameter_child.attrib["unit1"] + r" & "
            elif "description" in parameter_child.keys():
                p_2col = r"p{\dimexpr" + self.numb_cell + r" + " \
                    + self.unit_cell + r" + 2\tabcolsep}"
                return_value = r"{" + parameter_child.attrib["text"] + ":" \
                               + r"}  & \multicolumn{2}{" + p_2col + r"}{" \
                               + parameter_child.attrib["description"] \
                               + r" } & "
            else:  # \multicolumn{1}{l}
                return_value = r"{" + parameter_child.attrib["text"] + ":"
                return_value += r"} &  & { } &"

            return_value = return_value + \
                parameter_child.attrib["value"] + \
                r" & " + parameter_child.attrib["unit"]
            return self.__Wcell + " & " + return_value + r" \\"
        else:
            if (parameter_child.attrib["text"] == "alignment mark"
                    and parameter_child.attrib["value"] != "None"):
                return_value = self.__Wcell + " & " \
                    + r"{" + parameter_child.attrib["text"] + ":} &" \
                    + r"\multicolumn{2}{l}{L" \
                    + str(parameter_child.attrib["value"]) + r" } &&\\ "
                return_value += self.__Wcell + " &&&  " + \
                    r" & & \\"
                # r"\multicolumn{1}{l}{\includegraphics[height=1.5cm]{L" + str(parameter_child.attrib["value"]) + r".pdf}} & " + \
                # r"\multicolumn{2}{c}{\includegraphics[height=1.5cm]{L" + str(parameter_child.attrib["value"]) + r"mark.pdf}} \\"
            else:
                return_value = self.__Wcell + " & " \
                    + r"{" + parameter_child.attrib["text"] \
                    + r":} & \multicolumn{4}{l}{" \
                    + str(parameter_child.attrib["value"]) + r" } \\"
            return return_value

    def table_header(self):
        cell_width_numb = "table-format=5.1,table-column-width=" \
            + self.numb_cell
        cell_width_unit = "table-column-width=" + self.unit_cell
        print(r"""
\begin{tabularx}{\textwidth}{
    p{""" + self.unit_cell + r"""}
    X
    S[""" + cell_width_numb + r""",table-unit-alignment=right]
    s[""" + cell_width_unit + r""",table-unit-alignment=left]
    S[""" + cell_width_numb + r""",table-unit-alignment=right]
    s[""" + cell_width_unit + r""",table-unit-alignment=left]}
    %% see http://tex.stackexchange.com/questions/170637/restarting-rowcolors
    %% see http://tex.stackexchange.com/questions/58390/tables-colouring-odd-even-mixed-up
    %% \setcounter{rownum}{0} """)


if __name__ == '__main__':
    a = xmlPrettyTable2col(filename=sys.argv[1], classIx=1)
    # print (r"The operator time in total is $\sum$ OT", glob_ot, r"and the machine time in total is ")
    # print (r"$\sum$ MT", glob_mt, r".\\")

    a = xmlPrettyTable2col(filename=sys.argv[1], classIx=1, subclassIx=None)
    a.classtype = "G "
    a.steptype = "G "
    a.printtex()