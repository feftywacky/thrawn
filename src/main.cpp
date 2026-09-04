#include "globals.h"
#include "misc.h"
#include "position.h"
#include "transposition_table.h"
#include "uci.h"

int main() {
    init_all();
    uci_loop(pos);

    delete pos;
    delete tt;
    return 0;
}
