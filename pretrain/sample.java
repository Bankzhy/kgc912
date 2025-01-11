public class Test {
static String trim(final String ¢){
  final String[] rows=¢.split("\n");
  for (int i=0; i < rows.length; ++i)   rows[i]=trimAbsolute(rows[i],TRIM_THRESHOLD,TRIM_SUFFIX);
  return String.join("\n",rows);
}
}