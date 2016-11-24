Sub LeaveNumberInField()
    Selection.Fields.Item(1).Code.Text = Replace(Selection.Fields.Item(1).Code.Text, "\h", "\# 0 \h")
    Selection.Fields.Update
End Sub