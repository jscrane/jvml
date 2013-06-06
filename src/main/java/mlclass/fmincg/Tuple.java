package mlclass.fmincg;

public class Tuple<P, Q> {

    private final P first;
    private final Q second;

    public Tuple(P first, Q second) {
        this.first = first;
        this.second = second;
    }

    public P getFirst() {
        return first;
    }

    public Q getSecond() {
        return second;
    }
}
