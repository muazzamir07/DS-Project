#include"Train.h"
#include"Test.h"
int main()
{
	Train tr("D:\\image",  512);
	tr.flow();
	Test t("D:\\image");
	t.flow();
}
